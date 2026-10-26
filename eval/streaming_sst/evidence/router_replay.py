"""Open-loop replay of AutoTerm's context-similarity routing on the audited ten-talk run.

The generated translation of the audited AutoTerm run is held fixed. For every decoder
window t, the routing context is the concatenation of the previous w emitted chunks
(causal: window t is routed from chunks before it). Contexts and slice prototypes are
encoded with BGE-M3 (CLS pooling, 128 tokens, left truncation), matching
framework/agents/term_memory/context_similarity.py. Slices are ranked by cosine
similarity; no switch guards are applied, so the numbers describe raw similarity
ranking, which is what pinning and the multi-slice working set have to stabilise.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

DESCRIPTIONS = {
    "nlp": "NLP, speech, translation, ML, dataset, and benchmark terms.",
    "medicine": "Clinical, disease, drug, procedure, and biomedical terms.",
    "education": "School, teaching, curriculum, assessment, and learning terms.",
    "finance": "Market, instrument, accounting, monetary, and trading terms.",
    "legal": "Law, court, contract, regulation, and liability terms.",
    "environment": "Climate, ecology, conservation, pollution, and disaster terms.",
    "entertainment": "Film, television, music, gaming, and media terms.",
    "science": "Biology, chemistry, physics, experiments, and research terms.",
    "sports": "Teams, tournaments, athletes, matches, and competition terms.",
    "art": "Painting, sculpture, museums, architecture, and art-history terms.",
}
EN = {
    "nlp": "Natural language processing, speech translation, machine learning, software systems, servers, databases, and load balancing.",
    "medicine": "Healthcare, clinical medicine, patients, diagnosis, treatment, drugs, disease, and public health.",
    "education": "Education, schools, students, teachers, classrooms, curriculum, assessment, colleges, and universities.",
    "finance": "Finance, markets, banking, investment, accounting, business, brands, prices, customers, and the economy.",
    "legal": "Law, courts, criminal cases, lawyers, contracts, legislation, legal rights, judgments, and appeals.",
    "environment": "Environment, climate, ecology, pollution, conservation, floods, storms, natural disasters, forests, and rivers.",
    "entertainment": "Entertainment, films, television, music, video games, actors, directors, consoles, and popular media.",
    "science": "Science, biology, chemistry, physics, experiments, cells, organisms, species, molecules, and evolution.",
    "sports": "Sports, football, basketball, teams, players, coaches, matches, leagues, tournaments, and championships.",
    "art": "Art, painting, sculpture, museums, artists, portraits, tapestries, frescoes, galleries, and art history.",
}
ZH = {
    "nlp": "自然语言处理、语音翻译、机器学习、软件系统、服务器、数据库和负载均衡。",
    "medicine": "医疗保健、临床医学、患者、诊断、治疗、药物、疾病和公共卫生。",
    "education": "教育、学校、学生、教师、课堂、课程、评估、学院和大学。",
    "finance": "金融、市场、银行、投资、会计、商业、品牌、价格、顾客和经济。",
    "legal": "法律、法院、刑事案件、律师、合同、立法、法律权利、判决和上诉。",
    "environment": "环境、气候、生态、污染、保护、洪水、风暴、自然灾害、森林和河流。",
    "entertainment": "娱乐、电影、电视、音乐、电子游戏、演员、导演、游戏主机和流行媒体。",
    "science": "科学、生物学、化学、物理学、实验、细胞、生物、物种、分子和进化。",
    "sports": "体育、足球、篮球、球队、球员、教练、比赛、联赛、锦标赛和冠军。",
    "art": "艺术、绘画、雕塑、博物馆、艺术家、肖像、挂毯、壁画、美术馆和艺术史。",
}
# note (luojiaxuan): independent rewrites of each field with different word choices, used to probe wording sensitivity.
PARA_EN = {
    "nlp": "Computational linguistics: language models, text and speech processing, automatic translation, corpora, and evaluation benchmarks.",
    "medicine": "Medical care: hospitals, physicians, illnesses, symptoms, therapies, medication, and clinical studies.",
    "education": "Learning and teaching: pupils, lecturers, lessons, exams, degrees, and academic institutions.",
    "finance": "Money and business: stocks, banks, loans, companies, trade, revenue, and economic policy.",
    "legal": "The justice system: judges, trials, attorneys, statutes, lawsuits, verdicts, and regulations.",
    "environment": "The natural world and climate change: ecosystems, emissions, wildlife protection, droughts, and hurricanes.",
    "entertainment": "Popular culture: movies, TV series, songs, celebrities, gaming, and streaming media.",
    "science": "Natural sciences: laboratory research, genes, chemical reactions, physical laws, and living organisms.",
    "sports": "Athletics and competitive games: clubs, athletes, seasons, scores, cups, and coaching.",
    "art": "Fine arts: paintings, sculptures, exhibitions, art galleries, painters, and artistic movements.",
}
PARA_ZH = {
    "nlp": "计算语言学：语言模型、文本与语音处理、自动翻译、语料库和评测基准。",
    "medicine": "医疗：医院、医生、疾病、症状、疗法、用药和临床研究。",
    "education": "教与学：学生、讲师、课程、考试、学位和学术机构。",
    "finance": "货币与商业：股票、银行、贷款、公司、贸易、营收和经济政策。",
    "legal": "司法体系：法官、审判、律师、法规、诉讼、裁决和监管。",
    "environment": "自然界与气候变化：生态系统、排放、野生动物保护、干旱和飓风。",
    "entertainment": "流行文化：电影、电视剧、歌曲、明星、游戏和流媒体。",
    "science": "自然科学：实验室研究、基因、化学反应、物理定律和生物体。",
    "sports": "体育竞技：俱乐部、运动员、赛季、比分、杯赛和执教。",
    "art": "美术：绘画、雕塑、展览、画廊、画家和艺术流派。",
}
NAME = {d: [d.replace("nlp", "natural language processing"), z] for d, z in zip(
    EN, ["自然语言处理", "医学", "教育", "金融", "法律", "环境", "娱乐", "科学", "体育", "艺术"])}

VARIANTS = {
    "original": {d: [DESCRIPTIONS[d], EN[d], ZH[d]] for d in EN},
    "english_only": {d: [DESCRIPTIONS[d], EN[d]] for d in EN},
    "chinese_only": {d: [ZH[d]] for d in EN},
    "description_only": {d: [DESCRIPTIONS[d]] for d in EN},
    "paraphrase": {d: [PARA_EN[d], PARA_ZH[d]] for d in EN},
    "name_only": {d: NAME[d] for d in EN},
}

# note (luojiaxuan): subfield slices for the granularity test; families map back to the coarse gold domains.
SUBFIELDS = {
    "nlp": {
        "machine_translation": ("Machine translation, speech translation, translation quality, bilingual corpora, and interpreting.", "机器翻译、语音翻译、翻译质量、双语语料和口译。"),
        "speech_recognition": ("Speech recognition, acoustic models, spoken language, transcription, and speech synthesis.", "语音识别、声学模型、口语、转写和语音合成。"),
        "question_answering": ("Question answering, information retrieval, search, knowledge bases, and reading comprehension.", "问答、信息检索、搜索、知识库和阅读理解。"),
        "language_models": ("Large language models, pretraining, prompting, transformers, fine-tuning, and text generation.", "大语言模型、预训练、提示、Transformer、微调和文本生成。"),
        "syntax_semantics": ("Syntax, parsing, morphology, part-of-speech tagging, word meaning, and lexical semantics.", "句法、句法分析、形态学、词性标注、词义和词汇语义。"),
        "dialogue": ("Dialogue systems, conversational agents, chatbots, user intents, and response generation.", "对话系统、会话智能体、聊天机器人、用户意图和回复生成。"),
        "databases": ("Databases, SQL queries, data management, indexing, and transactions.", "数据库、SQL查询、数据管理、索引和事务。"),
        "software_engineering": ("Software engineering, programming, source code, testing, debugging, and software systems.", "软件工程、编程、源代码、测试、调试和软件系统。"),
    },
    "medicine": {
        "oncology": ("Oncology, cancer, tumors, chemotherapy, metastasis, and cancer staging.", "肿瘤学、癌症、肿瘤、化疗、转移和癌症分期。"),
        "radiation_therapy": ("Radiation therapy, radiotherapy planning, dose, linear accelerators, and target volumes.", "放射治疗、放疗计划、剂量、直线加速器和靶区。"),
        "cardiology": ("Cardiology, heart disease, blood pressure, arrhythmia, and cardiac surgery.", "心脏病学、心脏病、血压、心律失常和心脏手术。"),
        "neurology": ("Neurology, the brain, stroke, epilepsy, dementia, and nerves.", "神经病学、大脑、中风、癫痫、痴呆和神经。"),
        "infectious_disease": ("Infectious diseases, viruses, bacteria, vaccines, antibiotics, and epidemics.", "传染病、病毒、细菌、疫苗、抗生素和流行病。"),
        "surgery": ("Surgery, operations, anesthesia, surgical procedures, and recovery.", "外科、手术、麻醉、外科操作和术后恢复。"),
        "pharmacology": ("Pharmacology, drugs, dosage, side effects, clinical trials, and prescriptions.", "药理学、药物、剂量、副作用、临床试验和处方。"),
        "pediatrics": ("Pediatrics, children's health, newborns, growth, and childhood diseases.", "儿科、儿童健康、新生儿、生长发育和儿童疾病。"),
    },
}
CORRECT_MED_SUB = {"oncology", "radiation_therapy"}


def encode(model, tok, texts, device, batch=64):
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.inference_mode():
            h = model(**enc).last_hidden_state[:, 0]
        out.append(F.normalize(h.float(), dim=-1).cpu())
    return torch.cat(out).numpy()


def centroids(model, tok, protos, device):
    names = list(protos)
    flat, spans = [], []
    for n in names:
        spans.append((len(flat), len(flat) + len(protos[n])))
        flat.extend(protos[n])
    emb = encode(model, tok, flat, device)
    c = np.stack([emb[a:b].mean(0) for a, b in spans])
    return names, c / np.linalg.norm(c, axis=1, keepdims=True)


def ranking_metrics(scores, names, gold, valid, ks):
    order = np.argsort(-scores, axis=1)
    gold_idx = np.array([names.index(g) for g in gold])
    rank = np.argmax(order == gold_idx[:, None], axis=1)
    top1 = order[:, 0]
    flips = int((top1[1:] != top1[:-1]).sum())
    out = {"top1_acc": round(float((rank[valid] == 0).mean()), 4), "top1_changes": flips}
    for k in ks:
        sets = [frozenset(row[:k]) for row in order]
        changes = sum(a != b for a, b in zip(sets[1:], sets[:-1]))
        out[f"cover@{k}"] = round(float((rank[valid] < k).mean()), 4)
        out[f"set_changes@{k}"] = int(changes)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--generated", type=Path, help="generate_router_prototypes.py output; adds one variant per sample")
    args = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.truncation_side = "left"
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    recs = json.load(open(args.run, encoding="utf-8"))["records"]
    gold = [str(r["expected_domain"]).lower() for r in recs]
    texts = [str(r.get("text") or "") for r in recs]
    n = len(recs)
    result = {"windows": n, "device": device}

    ctx_emb = {}
    for w in (1, 2, 4, 8):
        ctx = ["".join(texts[max(0, t - w):t]) for t in range(n)]
        nonempty = np.array([bool(c.strip()) for c in ctx])
        emb = encode(model, tok, [c if c.strip() else " " for c in ctx], device)
        # note (luojiaxuan): an empty context keeps the previous window's embedding, as a live router would.
        for t in range(1, n):
            if not nonempty[t]:
                emb[t] = emb[t - 1]
        ctx_emb[w] = (emb, nonempty)

    valid = np.ones(n, dtype=bool)
    valid[0] = False
    variants = dict(VARIANTS)
    if args.generated:
        gen = json.loads(args.generated.read_text(encoding="utf-8"))["slices"]
        for i in range(len(next(iter(gen.values()))["samples"])):
            variants[f"generated_s{i}"] = {
                name.removesuffix("_core_1k"): [entry["samples"][i][k] for k in ("description", "en", "zh")]
                for name, entry in gen.items()
            }
    table = {}
    for vname, protos in variants.items():
        names, c = centroids(model, tok, protos, device)
        for w, (emb, _) in ctx_emb.items():
            table[f"{vname}|w={w}"] = ranking_metrics(emb @ c.T, names, gold, valid, ks=(1, 2, 3, 4, 6))
    result["coarse"] = table

    fine = {}
    for d, subs in SUBFIELDS.items():
        for s, (en, zh) in subs.items():
            fine[f"{d}:{s}"] = [en, zh]
    for d in EN:
        if d not in SUBFIELDS:
            fine[f"{d}:{d}"] = VARIANTS["original"][d]
    names, c = centroids(model, tok, fine, device)
    fam = np.array([n_.split(":")[0] for n_ in names])
    gran = {"slices": len(names)}
    for w, (emb, _) in ctx_emb.items():
        s = emb @ c.T
        order = np.argsort(-s, axis=1)
        row = {}
        for k in (1, 2, 4, 8):
            topk_fam = fam[order[:, :k]]
            fam_cover = np.array([g in topk_fam[t] for t, g in enumerate(gold)])
            purity = np.array([(topk_fam[t] == g).mean() for t, g in enumerate(gold)])
            row[f"family_cover@{k}"] = round(float(fam_cover[valid].mean()), 4)
            row[f"gold_family_share_of_top{k}"] = round(float(purity[valid].mean()), 4)
        med = np.array([g == "medicine" for g in gold]) & valid
        med_names = [i for i, n_ in enumerate(names) if n_.startswith("medicine:")]
        best_med = np.array([names[med_names[np.argmax(s[t, med_names])]].split(":")[1] for t in range(n)])
        row["medicine_windows_best_medicine_subslice_is_oncology_or_radiotherapy"] = round(float(np.isin(best_med[med], list(CORRECT_MED_SUB)).mean()), 4)
        top1_names = np.array([names[i] for i in order[:, 0]])
        row["medicine_windows_top1_is_oncology_or_radiotherapy"] = round(float(np.isin(top1_names[med], [f"medicine:{x}" for x in CORRECT_MED_SUB]).mean()), 4)
        gran[f"w={w}"] = row
    result["granularity"] = gran
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
