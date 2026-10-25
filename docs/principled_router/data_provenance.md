# 数据来源与许可核对(2026-10-24 PT)

## ESO 医学讲座(medicine 评测的 5 个 talk)

- 来源:European School of Oncology(ESO)的 e-learning 讲座。原始数据在 Taurus `/mnt/data1/jiaxingxu/eso_v1.0/{dev,train,test}/<id>/`,含 `<id>.m4a`、`manual_transcription/sentence_segmented/ie<id>.srt`,部分样本有 `manual_translations/{de,es,fr,sl}/`(`.lst` 时间戳 + `.de` 译文 + `.en` 原文,另有 `.docx`)。
- 评测用的 5 个 talk(404、606、545006、596001、605000)都在 `test/`,都带 `manual_translations/de`(行数 170 / 266 / 144 / 339 / 480)。全库 `train` 990 讲、`dev` 6、`test` 6;带 `manual_translations/de` 的共 10 个。
- **参考译文出处(已核对处理脚本 `/home/jiaxingxu/rag-sst/eso-dataset/README.md` 与 `eso_pipeline_v1.py`)**:
  - **de:数据集自带的人工翻译**(`manual_translations/de/<id>.de`)。处理脚本把它按 `.lst` 时间戳和文本对齐回 SRT 句子,并强制术语的德语译文直接取自这句人工译文。注意两点:对齐是自动的,句子切分与原始译文分段不一一对应;处理 README 有一处把它称作 "de 的 sentence-level MT",但目录名与 README 其余部分都写"人工翻译",原始数据集没有找到 dataset card 能确认它是纯人工还是译后编辑。
  - **zh:Gemini 生成**(v1 pipeline,整条 sample 一次性翻译并抽术语)。
  - **ja:GPT 生成**(v2 pipeline,经 OpenRouter 的 OpenAI-compatible 接口)。
  - 结论与用户说法、论文附录(`demo_paper_emnlp/latex/sections/8_appendix.tex` 第 200 行)一致:德语人工,中文与日语为模型生成。
- **许可:未找到**。`eso_v1.0` 根目录与样本目录下没有 LICENSE / README。公开发布任何派生数据(子专科 slice、切好的音频)前必须先向数据提供方(jiaxingxu)确认来源协议;在此之前只在私有存储内使用。

## ACL 60/60

- 公开镜像 `ymoslem/acl-6060`(HF,CC BY 4.0):dev 468 句、eval 416 句,含 en/ar/de/fa/fr/ja/nl/pt/ru/tr/zh 参考,参考为人工翻译(IWSLT 2023 共享任务)。镜像不带 talk id,talk 划分需从原始发布取。
- 十 talk run 里的 5 个 ACL talk 就是 dev 的 5 篇:2022.acl-long.110(在线语义解析/任务型对话)、117(检索引导的 QA 反事实生成)、268(西班牙语未同化借词检测语料)、367(KinyaBERT,形态感知语言模型)、590(可训练表示池化的稀疏 Transformer)。标题取自 ACL Anthology bib。
- 术语 gold 已在仓库:`eval/streaming_sst/acl_gold_technical.json` 等;de/ja 版本见论文附录的多语言节。
