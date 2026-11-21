"""Domain taxonomy for adaptive working-glossary defaults.

Runtime routing is window-topic-first by default. The keyword lists here provide
the high-precision source/ASR topic signal, plus offline working-slice ranking
seeds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

AUTO_WORKING_PRESET = "auto_working"

GENERAL_DOMAIN = "general"
COMMON_WORKING_PRESET = "common_10k"
REALSI_DOMAIN_TO_WORKING_DOMAIN: Dict[str, str] = {
    "technology": "nlp",
    "healthcare": "medicine",
    "education": "education",
    "finance": "finance",
    "law": "legal",
    "environment": "environment",
    "entertainment": "entertainment",
    "science": "science",
    "sports": "sports",
    "art": "art",
}
DOMAIN_TO_PRESET: Dict[str, str] = {
    "nlp": "nlp_core_10k",
    "medicine": "medicine_core_10k",
    "education": "education_core_10k",
    "finance": "finance_core_10k",
    "legal": "legal_core_10k",
    "environment": "environment_core_10k",
    "entertainment": "entertainment_core_10k",
    "science": "science_core_10k",
    "sports": "sports_core_10k",
    "art": "art_core_10k",
}

WORKING_GLOSSARY_PRESETS = (COMMON_WORKING_PRESET, *DOMAIN_TO_PRESET.values())
WORKING_DOMAINS = tuple(DOMAIN_TO_PRESET.keys())

DOMAIN_ROUTER_PROTOTYPES: Dict[str, Tuple[str, ...]] = {
    "nlp": (
        "Natural language processing, speech translation, machine learning, software systems, servers, databases, and load balancing.",
        "自然语言处理、语音翻译、机器学习、软件系统、服务器、数据库和负载均衡。",
    ),
    "medicine": (
        "Healthcare, clinical medicine, patients, diagnosis, treatment, drugs, disease, and public health.",
        "医疗保健、临床医学、患者、诊断、治疗、药物、疾病和公共卫生。",
    ),
    "education": (
        "Education, schools, students, teachers, classrooms, curriculum, assessment, colleges, and universities.",
        "教育、学校、学生、教师、课堂、课程、评估、学院和大学。",
    ),
    "finance": (
        "Finance, markets, banking, investment, accounting, business, brands, prices, customers, and the economy.",
        "金融、市场、银行、投资、会计、商业、品牌、价格、顾客和经济。",
    ),
    "legal": (
        "Law, courts, criminal cases, lawyers, contracts, legislation, legal rights, judgments, and appeals.",
        "法律、法院、刑事案件、律师、合同、立法、法律权利、判决和上诉。",
    ),
    "environment": (
        "Environment, climate, ecology, pollution, conservation, floods, storms, natural disasters, forests, and rivers.",
        "环境、气候、生态、污染、保护、洪水、风暴、自然灾害、森林和河流。",
    ),
    "entertainment": (
        "Entertainment, films, television, music, video games, actors, directors, consoles, and popular media.",
        "娱乐、电影、电视、音乐、电子游戏、演员、导演、游戏主机和流行媒体。",
    ),
    "science": (
        "Science, biology, chemistry, physics, experiments, cells, organisms, species, molecules, and evolution.",
        "科学、生物学、化学、物理学、实验、细胞、生物、物种、分子和进化。",
    ),
    "sports": (
        "Sports, football, basketball, teams, players, coaches, matches, leagues, tournaments, and championships.",
        "体育、足球、篮球、球队、球员、教练、比赛、联赛、锦标赛和冠军。",
    ),
    "art": (
        "Art, painting, sculpture, museums, artists, portraits, tapestries, frescoes, galleries, and art history.",
        "艺术、绘画、雕塑、博物馆、艺术家、肖像、挂毯、壁画、美术馆和艺术史。",
    ),
}

WORKING_PRESET_META: Dict[str, Dict[str, str]] = {
    AUTO_WORKING_PRESET: {
        "label": "Automatic terminology",
        "domain": "auto",
        "description": "Routes directly among domain-specific terminology slices.",
    },
    "common_10k": {
        "label": "Common-terms diagnostic",
        "domain": GENERAL_DOMAIN,
        "description": "Diagnostic common-terms slice; not active by default.",
    },
    "nlp_core_10k": {
        "label": "NLP core working glossary 10k",
        "domain": "nlp",
        "description": "NLP, speech, translation, ML, dataset, and benchmark terms.",
    },
    "medicine_core_10k": {
        "label": "Medicine core working glossary 10k",
        "domain": "medicine",
        "description": "Clinical, disease, drug, procedure, and biomedical terms.",
    },
    "education_core_10k": {
        "label": "Education core working glossary 10k",
        "domain": "education",
        "description": "School, teaching, curriculum, assessment, and learning terms.",
    },
    "finance_core_10k": {
        "label": "Finance core working glossary 10k",
        "domain": "finance",
        "description": "Market, instrument, accounting, monetary, and trading terms.",
    },
    "legal_core_10k": {
        "label": "Legal core working glossary 10k",
        "domain": "legal",
        "description": "Law, court, contract, regulation, and liability terms.",
    },
    "environment_core_10k": {
        "label": "Environment core working glossary 10k",
        "domain": "environment",
        "description": "Climate, ecology, conservation, pollution, and disaster terms.",
    },
    "entertainment_core_10k": {
        "label": "Entertainment core working glossary 10k",
        "domain": "entertainment",
        "description": "Film, television, music, gaming, and media terms.",
    },
    "science_core_10k": {
        "label": "Science core working glossary 10k",
        "domain": "science",
        "description": "Biology, chemistry, physics, experiments, and research terms.",
    },
    "sports_core_10k": {
        "label": "Sports core working glossary 10k",
        "domain": "sports",
        "description": "Teams, tournaments, athletes, matches, and competition terms.",
    },
    "art_core_10k": {
        "label": "Art core working glossary 10k",
        "domain": "art",
        "description": "Painting, sculpture, museums, architecture, and art-history terms.",
    },
}

DOMAIN_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "nlp": (
        "language model",
        "large language model",
        "natural language processing",
        "machine translation",
        "speech translation",
        "speech recognition",
        "simultaneous interpretation",
        "retrieval augmented generation",
        "rag",
        "token",
        "tokenizer",
        "benchmark",
        "dataset",
        "transformer",
        "encoder",
        "decoder",
        "alignment",
        "bleu",
        "comet",
        "attention",
        "embedding",
        "fine-tuning",
        "pretraining",
        "corpus",
        "treebank",
        "named entity",
        "question answering",
        "summarization",
        "information retrieval",
        "load balancing",
        "distributed system",
        "computer network",
        "database",
        "cloud computing",
        "software engineering",
        "application programming interface",
        "data replication",
    ),
    "medicine": (
        "patient",
        "clinical",
        "disease",
        "diagnosis",
        "treatment",
        "medicine",
        "drug",
        "protein",
        "cancer",
        "vaccine",
        "hospital",
        "trial",
        "therapy",
        "symptom",
        "infection",
        "syndrome",
        "tumor",
        "surgery",
        "pharmacology",
        "cardiology",
        "oncology",
        "immunology",
        "healthcare",
        "health care",
        "doctor",
        "clinic",
        "antibody",
        "genetics",
        "molecular medicine",
        "digital health",
    ),
    "education": (
        "education",
        "school",
        "student",
        "teacher",
        "classroom",
        "curriculum",
        "pedagogy",
        "university",
        "college",
        "kindergarten",
        "literacy",
        "assessment",
        "tuition",
        "scholarship",
        "academic degree",
        "vocational training",
        "distance learning",
        "special education",
        "educational psychology",
    ),
    "finance": (
        "finance",
        "financial",
        "market",
        "stock",
        "revenue",
        "valuation",
        "equity",
        "bond",
        "interest rate",
        "earnings",
        "etf",
        "inflation",
        "monetary",
        "trading",
        "portfolio",
        "derivative",
        "dividend",
        "asset",
        "liability",
        "cash flow",
        "central bank",
        "treasury",
        "bank",
        "banking",
        "company",
        "corporation",
        "business",
        "economy",
        "economic",
        "accounting",
        "investment",
        "investor",
        "fund",
        "insurance",
        "mortgage",
        "loan",
        "credit",
        "currency",
        "exchange rate",
        "taxation",
        "budget",
        "capital market",
        "commodity",
        "retail",
        "marketing",
        "brand",
        "consumer",
    ),
    "legal": (
        "law",
        "court",
        "regulation",
        "contract",
        "liability",
        "policy",
        "statute",
        "legal",
        "lawsuit",
        "plaintiff",
        "defendant",
        "jurisdiction",
        "compliance",
        "patent",
        "copyright",
        "tort",
        "appeal",
        "arbitration",
        "clause",
        "attorney",
        "lawyer",
        "criminal",
        "crime",
        "murder",
        "legislation",
        "treaty",
        "constitution",
        "judicial",
        "judge",
        "prosecutor",
        "prison",
        "legal right",
    ),
    "environment": (
        "climate",
        "environment",
        "ecology",
        "ecosystem",
        "pollution",
        "emission",
        "carbon",
        "biodiversity",
        "conservation",
        "renewable energy",
        "flood",
        "drought",
        "hurricane",
        "wildfire",
        "weather",
        "recycling",
        "sustainability",
        "greenhouse gas",
        "sea level",
        "natural disaster",
        "wildlife",
        "forest",
        "ocean",
        "river",
        "water resource",
        "waste management",
        "endangered species",
        "habitat",
        "storm",
        "cyclone",
        "earthquake",
        "coastal",
        "floodplain",
        "sewage",
        "protected area",
        "national park",
        "environmental policy",
    ),
    "entertainment": (
        "entertainment",
        "film",
        "movie",
        "television",
        "video game",
        "gameplay",
        "music",
        "album",
        "actor",
        "actress",
        "cinema",
        "animation",
        "streaming media",
        "game console",
        "screenplay",
        "box office",
        "record label",
        "broadcasting",
    ),
    "science": (
        "science",
        "scientific",
        "biology",
        "chemistry",
        "physics",
        "astronomy",
        "geology",
        "experiment",
        "hypothesis",
        "laboratory",
        "organism",
        "species",
        "evolution",
        "cell biology",
        "molecule",
        "particle",
        "theory",
        "research institute",
        "microscope",
        "taxonomy",
    ),
    "sports": (
        "sport",
        "football",
        "soccer",
        "basketball",
        "baseball",
        "tennis",
        "cricket",
        "athlete",
        "tournament",
        "championship",
        "league",
        "world cup",
        "olympic",
        "coach",
        "stadium",
        "match",
        "team",
        "goalkeeper",
        "medal",
        "referee",
    ),
    "art": (
        "art",
        "painting",
        "painter",
        "sculpture",
        "sculptor",
        "museum",
        "gallery",
        "artist",
        "portrait",
        "fresco",
        "tapestry",
        "canvas",
        "engraving",
        "renaissance",
        "architecture",
        "ceramic",
        "mural",
        "visual arts",
        "art history",
        "masterpiece",
    ),
}

ENTITY_TYPE_DOWNRANK = (
    "human",
    "person",
    "given name",
    "family name",
    "place",
    "country",
    "city",
    "geographic",
    "wikimedia",
    "category",
    "disambiguation",
)


@dataclass(frozen=True)
class DomainScore:
    domain: str
    score: float
    reason: str


@dataclass(frozen=True)
class TopicKeyword:
    pattern: str
    domain: str
    weight: float = 1.0
    case_sensitive: bool = False


DOMAIN_TOPIC_KEYWORDS: Dict[str, Tuple[TopicKeyword, ...]] = {
    "nlp": (
        TopicKeyword(r"\blanguage model(s)?\b", "nlp", 1.2),
        TopicKeyword(r"\blarge language model(s)?\b", "nlp", 1.3),
        TopicKeyword(r"\bnatural language processing\b", "nlp", 1.4),
        TopicKeyword(r"\bNLP\b", "nlp", 1.2, case_sensitive=True),
        TopicKeyword(r"\bBERT\b", "nlp", 1.2, case_sensitive=True),
        TopicKeyword(r"\btransformer(s)?\b", "nlp", 1.0),
        TopicKeyword(r"\bencoder(s)?\b", "nlp", 0.8),
        TopicKeyword(r"\bdecoder(s)?\b", "nlp", 0.8),
        TopicKeyword(r"\bdataset(s)?\b", "nlp", 0.7),
        TopicKeyword(r"\bbenchmark(s)?\b", "nlp", 0.8),
        TopicKeyword(r"\bcorpus|corpora\b", "nlp", 1.0),
        TopicKeyword(r"\bannotation(s)?\b", "nlp", 0.8),
        TopicKeyword(r"\bparser(s)?|parsing\b", "nlp", 1.0),
        TopicKeyword(r"\bmachine translation\b", "nlp", 1.2),
        TopicKeyword(r"\bentity recognition\b", "nlp", 1.1),
        TopicKeyword(r"\bdependency parsing\b", "nlp", 1.1),
        TopicKeyword(r"\bBLEU\b", "nlp", 1.1, case_sensitive=True),
        TopicKeyword(r"\battention\b", "nlp", 0.9),
        TopicKeyword(r"\bembedding(s)?\b", "nlp", 0.9),
        TopicKeyword(r"\btoken(s|ization|izer)?\b", "nlp", 0.9),
        TopicKeyword(r"\bpretraining|pre-trained|pretrained\b", "nlp", 1.0),
        TopicKeyword(r"\bfine[- ]?tuning\b", "nlp", 1.0),
        TopicKeyword(r"\bprompt(s|ing)?\b", "nlp", 0.8),
        TopicKeyword(r"\bload balancing|distributed system(s)?\b", "nlp", 1.1),
        TopicKeyword(r"\bserver(s)?|database(s)?|data replication\b", "nlp", 0.9),
        TopicKeyword(r"\bcloud computing|software engineering|API(s)?\b", "nlp", 1.0),
        TopicKeyword(r"语言模型", "nlp", 1.2),
        TopicKeyword(r"自然语言处理", "nlp", 1.4),
        TopicKeyword(r"机器翻译|语音翻译|同声传译", "nlp", 1.2),
        TopicKeyword(r"数据集|基准测试|语料库", "nlp", 1.0),
        TopicKeyword(r"标注|实体识别|依存句法", "nlp", 1.0),
        TopicKeyword(r"注意力|嵌入|预训练|微调", "nlp", 0.9),
        TopicKeyword(r"负载均衡|分布式系统|服务器|数据库", "nlp", 1.1),
        TopicKeyword(r"数据复制|云计算|软件工程|应用程序接口", "nlp", 1.0),
        TopicKeyword(r"言語モデル", "nlp", 1.2),
        TopicKeyword(r"自然言語処理", "nlp", 1.4),
        TopicKeyword(r"機械翻訳|音声翻訳|同時通訳", "nlp", 1.2),
        TopicKeyword(r"データセット|ベンチマーク|コーパス", "nlp", 1.0),
        TopicKeyword(r"アノテーション|固有表現|構文解析", "nlp", 1.0),
        TopicKeyword(r"注意機構|埋め込み|事前学習|ファインチューニング", "nlp", 0.9),
        TopicKeyword(r"トランスフォーマー|トークン", "nlp", 0.9),
        TopicKeyword(r"\bSprachmodell(e|en)?\b", "nlp", 1.2),
        TopicKeyword(r"\bSprachverarbeitung\b", "nlp", 1.4),
        TopicKeyword(r"\bmaschinelle[nr]? (\u00dcbersetzung|Uebersetzung)\b", "nlp", 1.2),
        TopicKeyword(r"\bDatens(a|\u00e4)tz(e|en)?\b|\bKorpus\b|\bKorpora\b", "nlp", 1.0),
        TopicKeyword(r"\bAnnotation(en)?\b|\bEinbettung(en)?\b", "nlp", 0.9),
        TopicKeyword(r"\bvortrainiert\w*\b|\bFeinabstimmung\b", "nlp", 1.0),
    ),
    "medicine": (
        TopicKeyword(r"\bpatient(s)?\b", "medicine", 1.0),
        TopicKeyword(r"\bclinical\b", "medicine", 1.1),
        TopicKeyword(r"\bdiagnos(is|es|tic)\b", "medicine", 1.1),
        TopicKeyword(r"\bsymptom(s)?\b", "medicine", 1.0),
        TopicKeyword(r"\bdisease(s)?\b", "medicine", 1.0),
        TopicKeyword(r"\bfever(s)?\b", "medicine", 1.0),
        TopicKeyword(r"\bheadache(s)?\b", "medicine", 1.0),
        TopicKeyword(r"\btablet(s)?\b", "medicine", 1.0),
        TopicKeyword(r"\bdose(s)?\b", "medicine", 1.0),
        TopicKeyword(r"\bmg\b", "medicine", 0.9),
        TopicKeyword(r"\btreatment(s)?\b", "medicine", 1.0),
        TopicKeyword(r"\bprescrib(e|ed|es|ing)\b", "medicine", 1.1),
        TopicKeyword(r"\bdrug(s)?\b", "medicine", 1.0),
        TopicKeyword(r"\bmedicine(s)?\b", "medicine", 1.0),
        TopicKeyword(r"\bdiabetes\b", "medicine", 1.2),
        TopicKeyword(r"\bhypertension\b", "medicine", 1.2),
        TopicKeyword(r"\bcancer(s)?\b", "medicine", 1.2),
        TopicKeyword(r"\btrial(s)?\b", "medicine", 0.9),
        TopicKeyword(r"\binfection(s)?\b", "medicine", 1.0),
        TopicKeyword(r"\bvaccine(s)?\b", "medicine", 1.1),
        TopicKeyword(r"\bMRI\b", "medicine", 1.1, case_sensitive=True),
        TopicKeyword(r"\bCT\b", "medicine", 0.9, case_sensitive=True),
        TopicKeyword(r"\bblood pressure\b", "medicine", 1.2),
        TopicKeyword(r"\bheart rate\b", "medicine", 1.1),
        TopicKeyword(r"\bsurger(y|ies|ical)\b", "medicine", 1.1),
        TopicKeyword(r"\boncolog(y|ical|ist|ists)\b", "medicine", 1.2),
        TopicKeyword(r"\bhospital(s)?\b", "medicine", 1.0),
        TopicKeyword(r"\bhealth ?care|digital health\b", "medicine", 1.2),
        TopicKeyword(r"\bdoctor(s)?|physician(s)?|clinic(s)?\b", "medicine", 1.0),
        TopicKeyword(r"\bantibod(y|ies)|immune system\b", "medicine", 1.1),
        TopicKeyword(r"\bCOVID(-19)?|coronavirus|pneumonia\b", "medicine", 1.2),
        TopicKeyword(r"\bgenetic(s)?|point-of-care testing\b", "medicine", 1.0),
        TopicKeyword(r"患者|病人", "medicine", 1.0),
        TopicKeyword(r"患者|臨床|診断", "medicine", 1.1),
        TopicKeyword(r"症状|疾患|治療", "medicine", 1.0),
        TopicKeyword(r"がん|腫瘍|化学療法|放射線", "medicine", 1.3),
        TopicKeyword(r"手術|投与|服用|医師", "medicine", 1.0),
        TopicKeyword(r"\bPatient(en|in|innen)?\b", "medicine", 1.0),
        TopicKeyword(r"\bklinisch\w*\b|\bDiagnose(n)?\b", "medicine", 1.1),
        TopicKeyword(r"\bSymptom(e|en)?\b|\bKrankheit(en)?\b", "medicine", 1.0),
        TopicKeyword(r"\bBehandlung(en)?\b|\bTherapie(n)?\b|\bMedikament(e|en)?\b", "medicine", 1.0),
        TopicKeyword(r"\bKrebs\b|\bTumor(e|en)?\b", "medicine", 1.3),
        TopicKeyword(r"\bChemotherapie\b|\bBestrahlung\b|\bStrahlentherapie\b", "medicine", 1.3),
        TopicKeyword(r"\bOperation(en)?\b|\bDosis\b|\bDosierung\b", "medicine", 1.0),
        TopicKeyword(r"\b(Arzt|\u00c4rzte|Onkolog(e|en|in|innen)?)\b", "medicine", 1.1),
        TopicKeyword(r"临床|诊断|症状|疾病", "medicine", 1.1),
        TopicKeyword(r"治疗|处方|药物|医学|医院", "medicine", 1.0),
        TopicKeyword(r"糖尿病|高血压|癌症|肿瘤|感染|疫苗", "medicine", 1.2),
        TopicKeyword(r"临床试验|手术|血压|心率|剂量|毫克", "medicine", 1.0),
        TopicKeyword(r"核磁|磁共振|CT", "medicine", 0.9),
        TopicKeyword(r"医疗保健|数字医疗|健康状况", "medicine", 1.2),
        TopicKeyword(r"医生|诊所|抗体|免疫系统", "medicine", 1.1),
        TopicKeyword(r"新冠|冠状病毒|肺炎|遗传学|即时检验", "medicine", 1.1),
    ),
    "education": (
        TopicKeyword(r"\beducation(al)?\b", "education", 1.2),
        TopicKeyword(r"\bschool(s|ing)?\b", "education", 1.1),
        TopicKeyword(r"\bstudent(s)?\b", "education", 1.0),
        TopicKeyword(r"\bteacher(s)?\b", "education", 1.0),
        TopicKeyword(r"\bcurriculum|curricula\b", "education", 1.2),
        TopicKeyword(r"\bclassroom(s)?\b", "education", 1.0),
        TopicKeyword(r"\bkindergarten\b", "education", 1.1),
        TopicKeyword(r"\bprimary school|secondary school|high school\b", "education", 1.2),
        TopicKeyword(r"\bcollege(s)?|universit(y|ies)\b", "education", 0.9),
        TopicKeyword(r"\bpedagog(y|ical)\b", "education", 1.2),
        TopicKeyword(r"\btuition|scholarship(s)?\b", "education", 1.0),
        TopicKeyword(r"教育|学校|学生|教师|老师", "education", 1.1),
        TopicKeyword(r"小学|中学|高中|大学|学前教育", "education", 1.2),
        TopicKeyword(r"课程|课堂|教学|升学|学位", "education", 1.0),
        TopicKeyword(r"学费|奖学金|教育体系", "education", 1.1),
    ),
    "finance": (
        TopicKeyword(r"\bmarket(s)?\b", "finance", 1.0),
        TopicKeyword(r"\bstock(s)?\b", "finance", 1.0),
        TopicKeyword(r"\brevenue\b", "finance", 1.0),
        TopicKeyword(r"\bvaluation(s)?\b", "finance", 1.0),
        TopicKeyword(r"\bequity\b", "finance", 1.0),
        TopicKeyword(r"\bbond(s)?\b", "finance", 1.0),
        TopicKeyword(r"\binterest rate(s)?\b", "finance", 1.1),
        TopicKeyword(r"\bearnings\b", "finance", 1.0),
        TopicKeyword(r"\binflation\b", "finance", 1.0),
        TopicKeyword(r"\btrading\b", "finance", 1.0),
        TopicKeyword(r"\bportfolio(s)?\b", "finance", 1.0),
        TopicKeyword(r"\bderivative(s)?\b", "finance", 1.0),
        TopicKeyword(r"\bdividend(s)?\b", "finance", 1.0),
        TopicKeyword(r"\bcash flow\b", "finance", 1.1),
        TopicKeyword(r"\bcentral bank(s)?\b", "finance", 1.1),
        TopicKeyword(r"\btreasury\b", "finance", 0.9),
        TopicKeyword(r"\bbrand(s)?\b", "finance", 0.9),
        TopicKeyword(r"\bcustomer(s)?|consumer(s)?\b", "finance", 0.8),
        TopicKeyword(r"\bbusiness(es)?\b", "finance", 0.8),
        TopicKeyword(r"\bprice(s|d|ing)?\b", "finance", 0.8),
        TopicKeyword(r"市场|股票|股价|证券", "finance", 1.0),
        TopicKeyword(r"收入|营收|估值|股权", "finance", 1.0),
        TopicKeyword(r"债券|利率|收益率|通胀", "finance", 1.1),
        TopicKeyword(r"交易|投资组合|衍生品|股息", "finance", 1.0),
        TopicKeyword(r"现金流|央行|中央银行|财政部", "finance", 1.0),
        TopicKeyword(r"品牌|顾客|客户|消费者|商业模式", "finance", 0.9),
        TopicKeyword(r"价格|收费|价值|商品", "finance", 0.8),
    ),
    "legal": (
        TopicKeyword(r"\blaw(s)?\b", "legal", 1.0),
        TopicKeyword(r"\bcourt(s)?\b", "legal", 1.0),
        TopicKeyword(r"\bregulation(s)?\b", "legal", 1.0),
        TopicKeyword(r"\bcontract(s)?\b", "legal", 1.0),
        TopicKeyword(r"\bliabilit(y|ies)\b", "legal", 1.0),
        TopicKeyword(r"\bstatute(s)?\b", "legal", 1.0),
        TopicKeyword(r"\blegal\b", "legal", 0.9),
        TopicKeyword(r"\blawsuit(s)?\b", "legal", 1.1),
        TopicKeyword(r"\bplaintiff(s)?\b", "legal", 1.1),
        TopicKeyword(r"\bdefendant(s)?\b", "legal", 1.1),
        TopicKeyword(r"\bjurisdiction(s)?\b", "legal", 1.0),
        TopicKeyword(r"\bcompliance\b", "legal", 0.9),
        TopicKeyword(r"\bpatent(s)?\b", "legal", 1.0),
        TopicKeyword(r"\bcopyright(s)?\b", "legal", 1.0),
        TopicKeyword(r"\barbitration\b", "legal", 1.0),
        TopicKeyword(r"\bmurder|criminal|crime(s)?\b", "legal", 1.1),
        TopicKeyword(r"\battorney(s)?|lawyer(s)?\b", "legal", 1.1),
        TopicKeyword(r"\bappeal(s|ed|ing)?\b", "legal", 1.0),
        TopicKeyword(r"\bconvict(ed|ion)?\b", "legal", 1.0),
        TopicKeyword(r"法律|法规|条例|监管", "legal", 1.0),
        TopicKeyword(r"法院|法庭|诉讼|起诉", "legal", 1.1),
        TopicKeyword(r"合同|协议|责任|赔偿", "legal", 1.0),
        TopicKeyword(r"原告|被告|管辖权|合规", "legal", 1.1),
        TopicKeyword(r"专利|版权|仲裁|判决", "legal", 1.0),
        TopicKeyword(r"谋杀|刑事|犯罪|辩护律师", "legal", 1.2),
        TopicKeyword(r"定罪|上诉|证人|起诉书", "legal", 1.0),
    ),
    "environment": (
        TopicKeyword(r"\bclimate( change)?\b", "environment", 1.2),
        TopicKeyword(r"\benvironment(al)?\b", "environment", 1.1),
        TopicKeyword(r"\bflood(s|ed|ing)?\b", "environment", 1.2),
        TopicKeyword(r"\bhurricane(s)?|wildfire(s)?|drought(s)?\b", "environment", 1.2),
        TopicKeyword(r"\bcarbon|emission(s)?|pollution\b", "environment", 1.1),
        TopicKeyword(r"\becosystem(s)?|biodiversity|conservation\b", "environment", 1.1),
        TopicKeyword(r"\bsustainab(le|ility)|renewable energy\b", "environment", 1.0),
        TopicKeyword(r"\bFEMA\b", "environment", 1.1, case_sensitive=True),
        TopicKeyword(r"气候|环境|生态|污染|排放", "environment", 1.1),
        TopicKeyword(r"洪水|飓风|野火|干旱|暴风雨", "environment", 1.2),
        TopicKeyword(r"碳排放|温室气体|生物多样性|可持续", "environment", 1.1),
        TopicKeyword(r"重建|防灾|自然灾害|建筑标准", "environment", 0.9),
    ),
    "entertainment": (
        TopicKeyword(r"\bvideo game(s)?|gameplay\b", "entertainment", 1.2),
        TopicKeyword(r"\bNintendo|Pok[eé]mon\b", "entertainment", 1.3, case_sensitive=True),
        TopicKeyword(r"\bfilm(s)?|movie(s)?|cinema\b", "entertainment", 1.1),
        TopicKeyword(r"\btelevision|TV show(s)?|series\b", "entertainment", 1.0),
        TopicKeyword(r"\bmusic|album(s)?|song(s)?|concert(s)?\b", "entertainment", 1.0),
        TopicKeyword(r"\bactor(s)?|actress(es)?|director(s)?\b", "entertainment", 1.0),
        TopicKeyword(r"\bconsole(s)?|gaming\b", "entertainment", 1.0),
        TopicKeyword(r"游戏|电子游戏|玩法|玩家", "entertainment", 1.2),
        TopicKeyword(r"任天堂|精灵宝可梦|原神", "entertainment", 1.3),
        TopicKeyword(r"电影|电视剧|综艺|动画|影院", "entertainment", 1.1),
        TopicKeyword(r"音乐|专辑|歌曲|演唱会|演员|导演", "entertainment", 1.0),
    ),
    "science": (
        TopicKeyword(r"\bscientist(s)?|scientific\b", "science", 1.1),
        TopicKeyword(r"\bbiology|chemistry|physics|astronomy|geology\b", "science", 1.2),
        TopicKeyword(r"\bscientific experiment(s)?|laboratory experiment(s)?\b", "science", 1.1),
        TopicKeyword(r"\bhypothes(is|es)\b", "science", 1.1),
        TopicKeyword(r"\borganism(s)?|species|evolution\b", "science", 1.1),
        TopicKeyword(r"\bcell(s|ular)?|molecule(s)?|particle(s)?\b", "science", 1.0),
        TopicKeyword(r"\bmicroscope|laborator(y|ies)\b", "science", 1.0),
        TopicKeyword(r"\bsponge(s)?|flagell(a|um|ates)?\b", "science", 1.2),
        TopicKeyword(r"科学|科研|假说", "science", 1.1),
        TopicKeyword(r"生物学|化学|物理学|天文学|地质学", "science", 1.2),
        TopicKeyword(r"细胞|分子|粒子|物种|生物|进化", "science", 1.1),
        TopicKeyword(r"海绵|鞭毛|显微镜|实验室", "science", 1.2),
    ),
    "sports": (
        TopicKeyword(r"\bfootball|soccer|basketball|baseball|tennis|cricket\b", "sports", 1.2),
        TopicKeyword(r"\bteam(s)?|player(s)?|athlete(s)?|coach(es)?\b", "sports", 0.9),
        TopicKeyword(r"\btournament(s)?|championship(s)?|league(s)?\b", "sports", 1.1),
        TopicKeyword(r"\bWorld Cup|Olympic(s)?\b", "sports", 1.2, case_sensitive=True),
        TopicKeyword(r"\bmatch(es)?|goal(s)?|stadium(s)?|referee(s)?\b", "sports", 1.0),
        TopicKeyword(r"足球|篮球|棒球|网球|板球", "sports", 1.2),
        TopicKeyword(r"球队|球员|运动员|教练|裁判", "sports", 1.0),
        TopicKeyword(r"世界杯|奥运会|联赛|锦标赛|杯赛", "sports", 1.2),
        TopicKeyword(r"比赛|进球|冠军|体育场", "sports", 1.0),
    ),
    "art": (
        TopicKeyword(r"\bpainting(s)?|painter(s)?|portrait(s)?\b", "art", 1.2),
        TopicKeyword(r"\bsculpture(s)?|sculptor(s)?|relief(s)?\b", "art", 1.2),
        TopicKeyword(r"\btapestr(y|ies)|fresco(es)?|engraving(s)?\b", "art", 1.2),
        TopicKeyword(r"\bmuseum(s)?|art gallery|artist(s)?\b", "art", 1.0),
        TopicKeyword(r"\bRenaissance|canvas(es)?|mural(s)?\b", "art", 1.1, case_sensitive=True),
        TopicKeyword(r"\bart history|visual art(s)?|masterpiece(s)?\b", "art", 1.1),
        TopicKeyword(r"绘画|画作|画家|肖像|油画", "art", 1.2),
        TopicKeyword(r"雕塑|雕刻|浮雕|挂毯|壁画", "art", 1.2),
        TopicKeyword(r"博物馆|美术馆|艺术家|艺术史", "art", 1.1),
        TopicKeyword(r"文艺复兴|画布|杰作|图像艺术", "art", 1.1),
    ),
}


def topic_keyword_scores(text: str) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    scores = {domain: 0.0 for domain in DOMAIN_TOPIC_KEYWORDS}
    hits: Dict[str, List[str]] = {domain: [] for domain in DOMAIN_TOPIC_KEYWORDS}
    if not text:
        return scores, hits
    for domain, keywords in DOMAIN_TOPIC_KEYWORDS.items():
        for keyword in keywords:
            flags = 0 if keyword.case_sensitive else re.IGNORECASE
            if re.search(keyword.pattern, text, flags):
                scores[domain] += float(keyword.weight)
                hits[domain].append(keyword.pattern)
    return scores, hits


def preset_for_domain(domain: str, default_preset: str = "none") -> str:
    return DOMAIN_TO_PRESET.get((domain or "").strip().lower(), default_preset)


def domain_for_preset(preset: str) -> str:
    for domain, candidate in DOMAIN_TO_PRESET.items():
        if candidate == preset:
            return domain
    return WORKING_PRESET_META.get(preset, {}).get("domain", GENERAL_DOMAIN)


def configured_working_presets(raw: str) -> Tuple[str, ...]:
    presets = tuple(p.strip() for p in (raw or "").split(",") if p.strip())
    return presets or WORKING_GLOSSARY_PRESETS


def keyword_hits(text: str, domain: str) -> int:
    blob = (text or "").lower()
    return sum(1 for kw in DOMAIN_KEYWORDS.get(domain, ()) if kw in blob)


def best_keyword_domain(text: str) -> DomainScore:
    scores = [
        DomainScore(domain=domain, score=float(keyword_hits(text, domain)), reason="keyword")
        for domain in DOMAIN_KEYWORDS
    ]
    scores.sort(key=lambda item: item.score, reverse=True)
    return scores[0] if scores else DomainScore(GENERAL_DOMAIN, 0.0, "none")


def entry_domain_score(row: Dict[str, object], domain: str) -> float:
    """Score one glossary row for a domain slice builder.

    This intentionally favors term-like technical phrases and down-ranks generic
    entity rows. It is a lightweight ranking signal, not a taxonomy.
    """

    term = str(row.get("term") or row.get("source_label") or "").strip()
    desc = str(row.get("short_description") or row.get("description") or "").strip()
    blob = f"{term} {desc}".lower()
    score = float(keyword_hits(blob, domain)) * 10.0
    if domain == GENERAL_DOMAIN:
        score = 1.0
    if " " in term.strip():
        score += 2.0
    if len(term) >= 8:
        score += 0.5
    rank = row.get("rank")
    if isinstance(rank, int):
        score += max(0.0, 3.0 - min(float(rank), 1_000_000.0) / 333_333.0)
    types: Iterable[object] = row.get("entity_types") if isinstance(row.get("entity_types"), list) else []
    type_blob = " ".join(str(t).lower() for t in types)
    if any(marker in type_blob or marker in blob for marker in ENTITY_TYPE_DOWNRANK):
        score -= 4.0
    return score
