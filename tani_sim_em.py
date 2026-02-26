import re
import math
from typing import Any, Dict, List, Optional, Tuple, Union

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.AllChem import GetMorganGenerator
# 禁用 RDKit 的日志输出，防止训练时控制台被刷屏
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

# --- 依赖库处理 (不再需要 fuzzy match 库，但为了兼容性保留 import 结构) ---
try:
    import Levenshtein
except ImportError:
    import difflib

# --- 核心工具函数 ---

def get_composite_fingerprint(smiles_str: str):
    """
    直接对点分隔的 SMILES 字符串（如 'A.B.C'）生成单一指纹。
    RDKit 会将其视为一个多片段分子处理。
    """
    if not smiles_str:
        return None
    try:
        # sanitize=True 会尝试标准化分子
        mol = Chem.MolFromSmiles(smiles_str, sanitize=True)
        if mol is None:
            return None
        
        # 使用 r=2, 2048bits, 启用手性以确保"精确匹配"涵盖立体化学
        gen = GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
        return gen.GetFingerprint(mol)
    except Exception:
        return None
    
def get_mol_fingerprint(mol: Chem.Mol):
    """生成 Morgan 指纹 (r=2, 2048bits, 包含手性信息)"""
    if mol is None:
        return None
    try:
        # 启用手性 includeChirality=True
        gen = GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True) 
        return gen.GetFingerprint(mol)
    except Exception:
        return None

def calculate_tanimoto(fp1, fp2) -> float:
    if fp1 is None or fp2 is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fp1, fp2)

# --- 评分逻辑类 ---

class IUPACScorer:
    @staticmethod
    def score(ground_truth: str, prediction: str) -> float:
        """
        IUPAC 完全匹配模式。
        仅当字符串完全一致时返回 1.0 (忽略首尾空格和大小写差异，以防模型输出格式微差)。
        如果需要极度严格的大小写匹配，请去掉 .lower()。
        """
        if not ground_truth or not prediction:
            return 0.0
            
        gt_clean = ground_truth.strip()
        pred_clean = prediction.strip()
        
        # 这里使用了 .lower() 忽略大小写差异，因为化学命名中大小写通常不改变物质本身
        # 如果你要求严格的大小写匹配，请改为: return 1.0 if gt_clean == pred_clean else 0.0
        return 1.0 if gt_clean.lower() == pred_clean.lower() else 0.0

class ReactionScorer:
    @staticmethod
    def parse_reaction_smiles(rxn_smiles: str) -> Optional[Dict[str, str]]:
        """
        解析 Reaction SMILES，兼容 A>B>C 和 A>>C
        """
        if not rxn_smiles: return None
        
        rxn_smiles = rxn_smiles.strip()

        # 1. 尝试标准 3 段式
        parts = rxn_smiles.split('>')
        if len(parts) == 3:
            return {
                'reactants': parts[0].strip(),
                'agents': parts[1].strip(),
                'products': parts[2].strip()
            }
        
        # 2. 尝试 2 段式 (无 agents)
        if ">>" in rxn_smiles:
            parts = rxn_smiles.split(">>")
            if len(parts) == 2:
                return {
                    'reactants': parts[0].strip(),
                    'agents': '',
                    'products': parts[1].strip()
                }
        
        return None

    @classmethod
    def score(cls, gt_smiles: str, pred_smiles: str) -> float:
        """
        计算反应精确匹配：
        要求 Reactants, Agents, Products 三部分的指纹相似度均为 1.0。
        只要有任何一部分不匹配，总分即为 0.0。
        """
        gt_clean = gt_smiles.strip()
        pred_clean = pred_smiles.strip()

        gt_parts = cls.parse_reaction_smiles(gt_clean)
        pred_parts = cls.parse_reaction_smiles(pred_clean)

        if gt_parts is None: return 0.0
        if pred_parts is None: return 0.0

        parts_to_check = ['products', 'reactants', 'agents']
        
        for name in parts_to_check:
            gt_comp = gt_parts.get(name, '')
            pred_comp = pred_parts.get(name, '')

            # 情况 A: 应该为空，但预测不为空 -> 失败
            if not gt_comp and pred_comp:
                return 0.0
            
            # 情况 B: 应该有内容，但预测为空 -> 失败
            if gt_comp and not pred_comp:
                return 0.0
            
            # 情况 C: 都有内容，检查指纹是否完全一致 (Sim == 1.0)
            if gt_comp and pred_comp:
                gt_fp = get_composite_fingerprint(gt_comp)
                pred_fp = get_composite_fingerprint(pred_comp)
                
                sim = calculate_tanimoto(gt_fp, pred_fp)
                
                # 如果指纹相似度不是 1.0，则视为不匹配，直接返回 0
                if sim < 1.0:
                    return 0.0
        
        # 如果所有部分都通过了检查
        return 1.0

# --- 主逻辑函数 ---

def format_reward(response: str) -> float:
    """
    检查格式。
    允许在标签前后有空白字符。
    """
    pattern = re.compile(r"\s*<think>.*?</think>\s*<answer>.*?</answer>\s*", re.DOTALL)
    match = re.fullmatch(pattern, response)
    return 1.0 if match else 0.0

def extract_content(text: str, tag: str) -> Optional[str]:
    """辅助函数：提取 XML 标签内容"""
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None

def accuracy_reward(response: str, ground_truth: str) -> float:
    """
    计算准确率奖励。
    现在强制执行 Binary Reward：只有完全匹配才得 1.0 分，否则 0.0 分。
    """
    try:
        # 1. 提取 <answer> 内容
        answer_content = extract_content(response, "answer")
        if answer_content is None:
            # Fallback
            answer_content = response.strip()

        # 2. 路由判断
        
        # Case A: IUPAC 命名任务
        if '<IUPAC>' in ground_truth:
            pred_iupac = extract_content(answer_content, "IUPAC")
            ground_truth_iupac = extract_content(ground_truth, "IUPAC")
            
            if pred_iupac is None: 
                return 0.0
            
            return IUPACScorer.score(ground_truth_iupac, pred_iupac)

        # Case B: 化学结构任务 (SMILES / Reaction)
        else:
            # 尝试提取 <SMILES>
            pred_smiles = extract_content(answer_content, "SMILES")
            if pred_smiles is None:
                return 0.0

            gt_clean = ground_truth.strip()

            # 判断是反应还是单分子
            if ">" in gt_clean:
                # Reaction Mode (Exact Match)
                return ReactionScorer.score(gt_clean, pred_smiles)
            else:
                # Molecule Mode (Exact Match based on Tanimoto=1)
                gt_mol = Chem.MolFromSmiles(gt_clean)
                pred_mol = Chem.MolFromSmiles(pred_smiles)
                
                if gt_mol is None: return 0.0
                if pred_mol is None: return 0.0
                
                fp_gt = get_mol_fingerprint(gt_mol)
                fp_pred = get_mol_fingerprint(pred_mol)
                
                sim = calculate_tanimoto(fp_gt, fp_pred)
                
                # 只有指纹完全一致才返回 1.0
                return 1.0 if sim == 1.0 else 0.0

    except Exception:
        return 0.0

def compute_score(reward_input: dict[str, Any], format_weight: float = 0.1) -> dict[str, float]:
    if not isinstance(reward_input, dict):
        raise ValueError("Please use `reward_type=sequential`.")

    format_score = format_reward(reward_input["response"])
    accuracy_score = accuracy_reward(reward_input["response"], reward_input["ground_truth"])
    
    return {
        "overall": round((1 - format_weight) * accuracy_score + format_weight * format_score, 5),
        "format": format_score,
        "accuracy": accuracy_score,
    }

# --- 测试用例 ---
if __name__ == '__main__':
    # 1. 测试 IUPAC (完全匹配)
    print("Test IUPAC (Match):")
    gt_iupac = "<IUPAC>4-hydroxy-3-methoxybenzaldehyde</IUPAC>"
    resp_iupac_correct = "<think>...</think><answer><IUPAC>4-Hydroxy-3-methoxybenzaldehyde</IUPAC></answer>" # 大小写差异在 chemistry 中通常被允许
    resp_iupac_wrong = "<think>...</think><answer><IUPAC>4-hydroxy-benzaldehyde</IUPAC></answer>"
    print(f"Correct: {compute_score({'response': resp_iupac_correct, 'ground_truth': gt_iupac})['accuracy']}")
    print(f"Wrong:   {compute_score({'response': resp_iupac_wrong, 'ground_truth': gt_iupac})['accuracy']}")

    # 2. 测试 Reaction SMILES (全匹配)
    print("\nTest Reaction SMILES (Exact Match):")
    gt_rxn = "CC(=O)O.OCC>>CC(=O)OCC.O"
    
    # 指纹 1.0 的预测 (顺序不同但在化学上等价)
    pred_rxn_correct = "OCC.CC(=O)O>>O.CC(=O)OCC" 
    resp_rxn_correct = f"<think>...</think><answer><SMILES>{pred_rxn_correct}</SMILES></answer>"
    
    # 缺少产物 (以前可能得部分分，现在应该是 0)
    pred_rxn_wrong = "OCC.CC(=O)O>>CC(=O)OCC" 
    resp_rxn_wrong = f"<think>...</think><answer><SMILES>{pred_rxn_wrong}</SMILES></answer>"
    
    print(f"Correct: {compute_score({'response': resp_rxn_correct, 'ground_truth': gt_rxn})['accuracy']}")
    print(f"Wrong:   {compute_score({'response': resp_rxn_wrong, 'ground_truth': gt_rxn})['accuracy']}")

    # 3. 测试 Molecule SMILES (Tanimoto=1)
    print("\nTest Molecule SMILES (Tanimoto=1):")
    gt_mol = "c1ccccc1" # Benzene
    
    # Kekule 式 (指纹应相同)
    pred_mol_correct = "C1=CC=CC=C1" 
    resp_mol_correct = f"<think>...</think><answer><SMILES>{pred_mol_correct}</SMILES></answer>"
    
    # 相似但不完全相同 (例如 Pyridine)
    pred_mol_similar = "c1ncccc1" 
    resp_mol_similar = f"<think>...</think><answer><SMILES>{pred_mol_similar}</SMILES></answer>"
    
    print(f"Correct (Benzene): {compute_score({'response': resp_mol_correct, 'ground_truth': gt_mol})['accuracy']}")
    print(f"Wrong (Pyridine):  {compute_score({'response': resp_mol_similar, 'ground_truth': gt_mol})['accuracy']}")