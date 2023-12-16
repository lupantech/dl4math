# Deep Learning for Mathematical Reasoning (DL4MATH)

[![Awesome](https://awesome.re/badge.svg)](https://github.com/lupantech/dl4math) 
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Survey](https://img.shields.io/badge/Survey-DL4MATH-blue)](https://github.com/lupantech/dl4math) 

This repository is the reading list on ***Deep Learning for Mathematical Reasoning (DL4MATH)***. 

**Contributors**: [Pan Lu](https://lupantech.github.io/) @UCLA, [Liang Qiu](https://www.lqiu.info/) @UCLA, [Wenhao Yu](https://wyu97.github.io/) @Notre Dame, [Sean Welleck](https://wellecks.com/) @UW, [Kai-Wei Chang](http://web.cs.ucla.edu/~kwchang/) @UCLA

For more details, please refer to the paper: [A Survey of Deep Learning for Mathematical Reasoning](https://arxiv.org/abs/2212.10535).

:bell: If you have any suggestions or notice something we missed, please don't hesitate to let us know. You can directly email Pan Lu (lupantech@gmail.com), comment on the [twitter](https://twitter.com/lupantech/status/1605400505697841155), or post an issue on this repo.



## 🧰 Resources

### Related Surveys

- **A Survey of Question Answering for Math and Science Problem**, arXiv:1705.04530 [[paper](https://arxiv.org/abs/1705.04530)]
- **The Gap of Semantic Parsing: A Survey on Automatic Math Word Problem Solvers**, TPAMI 2019 [[paper](https://arxiv.org/abs/1808.07290)]
- **Representing Numbers in NLP: a Survey and a Vision**, NACL 2021 [[paper](https://aclanthology.org/2021.naacl-main.53/)]
- **Survey on Mathematical Word Problem Solving Using Natural Language Processing**, ICIICT 2021 [[paper](https://ieeexplore.ieee.org/abstract/document/8741437)]
- **A Survey in Mathematical Language Processing**, arXiv:2205.15231 [[paper](https://arxiv.org/abs/2205.15231)]
- **Partial Differential Equations Meet Deep Neural Networks: A Survey**, arXiv:2211.05567 [[paper](https://arxiv.org/abs/2211.05567)]
- :fire: **Reasoning with Language Model Prompting: A Survey**, arXiv:2212.09597 [[paper](https://arxiv.org/abs/2212.09597)]
- :fire: **Towards Reasoning in Large Language Models**: arXiv:2212.10403 [[paper](https://arxiv.org/abs/2212.10403)]
- :fire: **A Survey for In-context Learning**, arXiv:2301.00234 [[paper](https://arxiv.org/abs/2301.00234)]

### Related Blogs

- :fire: **How does GPT Obtain its Ability? Tracing Emergent Abilities of Language Models to their Sources**, Dec 2022, Yao Fu’s Notion [[link](https://yaofu.notion.site/How-does-GPT-Obtain-its-Ability-Tracing-Emergent-Abilities-of-Language-Models-to-their-Sources-b9a57ac0fcf74f30a1ab9e3e36fa1dc1)]

### Workshops

- :fire: **The 1st MATH-AI Workshop: the Role of Mathematical Reasoning in General Artificial Intelligence**, ICLR 2021 [[website](https://mathai-iclr.github.io/)]
- :fire: **Math AI for Education: Bridging the Gap Between Research and Smart Education (MATHAI4ED)**, NeurIPS 2021 [[website](https://mathai4ed.github.io/)]
- :fire: **The 1st Workshop on Mathematical Natural Language Processing**, EMNLP 2022 [[website](https://sites.google.com/view/1st-mathnlp/)]
- :fire: **The 2nd MATH-AI Workshop: Toward Human-Level Mathematical Reasoning**, NeurIPS 2022  [[website](https://mathai2022.github.io/)]
- :fire: **FLAIM: Formal Languages, AI and Mathematics**, IHP & META 2022  [[YouTube](https://www.youtube.com/playlist?list=PLgBHexwnIcdueioZA-fgrx0dxXY2tJu6H)]
- :fire: **AI to Assist Mathematical Reasoning: A Workshop**, NASEM 2023  [[YouTube](https://www.youtube.com/playlist?list=PLgBHexwnIcdtAv9jVYnXAjCMKA0pNDXxJ)]

### Talks

- **Can GPT-3 do math? | Grant Sanderson and Lex Fridman**, 2020 [[YouTube](https://www.youtube.com/watch?v=TMxAbNAVrzI&ab_channel=LexClips)]
- **Computer Scientist Explains One Concept in 5 Levels of Difficulty**, 2022 [[YouTube](https://www.youtube.com/watch?v=fOGdb1CTu5c)]



## 🎨 Mathematical Reasoning Benchmarks

### Math Word Problems (MWP)

- [AI2/Verb395] **Learning to Solve Arithmetic Word Problems with Verb Categorization**, EMNLP 2014 [[paper](https://aclanthology.org/D14-1058/)]
- [Alg514] **Learning to automatically solve algebra word problems**, ACL 2014 [[paper](https://aclanthology.org/P14-1026/)]
- [IL] **Reasoning about Quantities in Natural Language**, TACL 2015 [[paper](https://aclanthology.org/Q15-1001/)]
- [SingleEQ] **Parsing Algebraic Word Problems into Equations**, TACL 2015 [[paper](https://aclanthology.org/Q15-1042/)]
- [DRAW] **Draw: A challenging and diverse algebra word problem set**, 2015 [[paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tech_rep.pdf)]
- [Dolphin1878] **Automatically solving number word problems by semantic parsing and reasoning**, EMNLP 2015 [[paper](https://aclanthology.org/D15-1135/)]
- [Dolphin18K] **How well do computers solve math word problems? large-scale dataset construction and evaluation**, ACL 2016 [[paper](https://aclanthology.org/P16-1084/)]
- [MAWPS] **MAWPS: A math word problem repository**, NAACL-HLT 2016 [[paper](https://aclanthology.org/N16-1136/)]
- [AllArith] **Unit dependency graph and its application to arithmetic word problem solving**, AAAI 2017 [[paper](https://arxiv.org/abs/1612.00969)]
- [DRAW-1K] **Annotating Derivations: A New Evaluation Strategy and Dataset for Algebra Word Problems**, ACL 2017 [[paper](https://aclanthology.org/E17-1047/)]
- :fire: [Math23K] **Deep neural solver for math word problems**, EMNLP 2017 [[paper](https://aclanthology.org/D17-1088/)]
- [AQuA] **Program Induction by Rationale Generation: Learning to Solve and Explain Algebraic Word Problems**, ACL 2017 [[paper](https://arxiv.org/abs/1705.04146)]
- [Aggregate] **Mapping to Declarative Knowledge for Word Problem Solving**, TACL 2018 [[paper](https://arxiv.org/abs/1712.09391)]
- :fire: [MathQA] **MathQA: Towards interpretable math word problem solving with operation-based formalisms**, NAACL-HLT 2019 [[paper](https://aclanthology.org/N19-1245/)]
- [ASDiv] **A Diverse Corpus for Evaluating and Developing English Math Word Problem Solvers**, ACL 2020 [[paper](https://arxiv.org/abs/2106.15772)]
- [HMWP] **Semantically-Aligned Universal Tree-Structured Solver for Math Word Problems**, EMNLP 2020 [[paper](https://arxiv.org/abs/2010.06823)]
- [Ape210K] **Ape210k: A large-scale and template-rich dataset of math word problems**, arXiv:2009.11506 [[paper](https://arxiv.org/abs/2009.11506)]
- :fire: [SVAMP] **Are NLP Models really able to Solve Simple Math Word Problems?**, NAACL-HIT 2021 [[paper](https://arxiv.org/abs/2103.07191)]
- :fire: [GSM8K] **Training verifiers to solve math word problems**, arXiv:2110.14168 [[paper](https://arxiv.org/abs/2110.14168)]
- :fire: [IconQA] **IconQA: A New Benchmark for Abstract Diagram Understanding and Visual Language Reasoning**, NeurIPS 2021] [[paper](https://arxiv.org/abs/2110.13214)]
- :fire: [MathQA-Python] **Program synthesis with large language models**, arXiv:2108.07732 [[paper](https://arxiv.org/abs/2108.07732)]
- [ArMATH] **ArMATH: a Dataset for Solving Arabic Math Word Problems**, LREC 2022 [[paper](https://aclanthology.org/2022.lrec-1.37/)]
- :fire: [TabMWP] **Dynamic Prompt Learning via Policy Gradient for Semi-structured Mathematical Reasoning**, arXiv:2209.14610, 2022 [[paper](https://arxiv.org/abs/2209.14610)]

### Theorem Proving (TP)

- [MML] **Four Decades of Mizar**, Journal of Automated Reasoning 2015, [[paper](https://dl.acm.org/doi/abs/10.1007/s10817-015-9345-1)]
- [HolStep] **HolStep: A Machine Learning Dataset for Higher-order Logic Theorem Proving**, ICLR 2017 [[paper](https://arxiv.org/abs/1703.00426)]
- [GamePad] **GamePad: A Learning Environment for Theorem Proving**, ICLR 2019 [[paper](https://arxiv.org/abs/1806.00608)]
- :fire: [CoqGym] **Learning to Prove Theorems via Interacting with Proof Assistants**, ICML 2019 [[paper](https://arxiv.org/abs/1905.09381)]
- [HOList] **HOList: An environment for machine learning of higher order logic theorem proving**, ICML 2019 [[paper](https://arxiv.org/abs/1904.03241)]
- [IsarStep] **IsarStep: a Benchmark for High-level Mathematical Reasoning**, ICLR 2021 [[paper](https://arxiv.org/abs/2006.09265)]
- [LISA] **LISA: Language models of ISAbelle proofs**, AITP 2021 [[paper](http://aitp-conference.org/2021/abstract/paper_17.pdf)]
- [INT] **INT: An Inequality Benchmark for Evaluating Generalization in Theorem Proving**, ICLR 2021 [[paper](https://arxiv.org/abs/2007.02924)]
- :fire: [NaturalProofs] **NaturalProofs: Mathematical Theorem Proving in Natural Language**, NeurIPS 2021 [[paper](https://arxiv.org/abs/2104.01112)]
- [NaturalProofs-Gen] **NaturalProver: Grounded Mathematical Proof Generation with Language Models**, NeurIPS 2022 [[paper](https://arxiv.org/abs/2205.12910)]
- :fire: [MiniF2F] **MiniF2F: a cross-system benchmark for formal Olympiad-level mathematics**, ICLR 2022 [[paper](https://arxiv.org/abs/2109.00110)]
- :fire: [LeanStep] **Proof Artifact Co-training for Theorem Proving with Language Models**, ICLR 2022 [[paper](https://arxiv.org/abs/2102.06203)]
- :fire: [miniF2F+informal] **Draft, Sketch, and Prove: Guiding Formal Theorem Provers with Informal Proofs**, arXiv:2210.12283 [[paper](https://arxiv.org/abs/2210.12283)]

### Geometry Problem Solving (GPS)

- :fire: [GEOS] **Solving geometry problems: Combining text and diagram interpretation**, EMNLP 2015 [[paper](https://aclanthology.org/D15-1171/)]
- [GeoShader] **Synthesis of solutions for shaded area geometry problems**, The Thirtieth International Flairs Conference, 2017 [[paper](https://www.aaai.org/ocs/index.php/FLAIRS/FLAIRS17/paper/viewFile/15416/14902)]
- [GEOS++] **From textbooks to knowledge: A case study in harvesting axiomatic knowledge from textbooks to solve geometry problems**, EMNLP 2017 [[paper](https://aclanthology.org/D17-1081/)]
- [GEOS-OS] **Learning to solve geometry problems from natural language demonstrations in textbooks**, Proceedings of the 6th Joint Conference on Lexical and Computational Semantics, 2017 [[paper](https://aclanthology.org/S17-1029/)]
- :fire: [Geometry3K] **Inter-GPS: Interpretable Geometry Problem Solving with Formal Language and Symbolic Reasoning**, ACL 2021 [[paper](https://aclanthology.org/2021.acl-long.528/)]
- [GeoQA] **GeoQA: A Geometric Question Answering Benchmark Towards Multimodal Numerical Reasoning**, Findings of ACL 2021 [[paper](https://aclanthology.org/2021.findings-acl.46.pdf)]
- [GeoQA+] **An Augmented Benchmark Dataset for Geometric Question Answering through Dual Parallel Text Encoding**, COLING 2022 [[paper](https://aclanthology.org/2022.coling-1.130/)]
- :fire: [UniGeo] **UniGeo: Unifying Geometry Logical Reasoning via Reformulating Mathematical Expression**, EMNLP 2022 [[paper](https://lupantech.github.io/papers/emnlp22_unigeo.pdf)]

### Math Question Answering (MathQA)

- [QUAREL] **QUAREL: A Dataset and Models for Answering Questions about Qualitative Relationships**, AAAI 2019 [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/4687)] 
- [McTaco] **“Going on a vacation” takes longer than “Going for a walk”: A Study of Temporal Commonsense Understanding**, EMNLP 2019 [[paper](https://aclanthology.org/D19-1332/)]
- :fire: [DROP] **DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs**, NAACL 2019 [[paper](https://aclanthology.org/N19-1246/)]
- :fire: [Mathematics] **Analysing Mathematical Reasoning Abilities of Neural Models**, ICLR 2019 [[paper](https://arxiv.org/abs/1904.01557)]
- [FinQA] **FinQA: A Dataset of Numerical Reasoning over Financial Data**, EMNLP 2021 [[paper](https://arxiv.org/abs/2109.00122)]
- [Fermi] **How Much Coffee Was Consumed During EMNLP 2019? Fermi Problems: A New Reasoning Challenge for AI**, EMNLP 2020 [[paper](https://arxiv.org/abs/2110.14207)]
- :fire: [MATH, AMPS] **Measuring Mathematical Problem Solving With the MATH Dataset**, NeurIPS 2021 [[paper](https://arxiv.org/abs/2103.03874)]
- [TAT-QA] **TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance**, ACL-JCNLP 2021 [[paper](https://arxiv.org/abs/2105.07624)]
- [MultiHiertt] **MultiHiertt: Numerical Reasoning over Multi Hierarchical Tabular and Textual Data**, ACL 2022 [[paper](https://aclanthology.org/2022.acl-long.454/)]
- [NumGLUE] **NumGLUE: A Suite of Fundamental yet Challenging Mathematical Reasoning Tasks**, ACL 2022 [[paper](https://aclanthology.org/2022.acl-long.246/)]
- :fire: [Lila] **Lila: A Unified Benchmark for Mathematical Reasoning**, EMNLP 2022 [[paper](https://arxiv.org/abs/2210.17517)]

### Other Quantitative Problems

- [FigureQA] **Figureqa: An annotated figure dataset for visual reasoning**, arXiv:1710.07300 [[paper](https://arxiv.org/abs/1710.07300)]
- :fire: [DVQA] **Dvqa: Understanding data visualizations via question answering**, CVPR 2018 [[paper](https://arxiv.org/abs/1801.08163)]
- [DREAM] **DREAM: A Challenge Dataset and Models for Dialogue-Based Reading Comprehension**,TACL 2019 [[paper](https://arxiv.org/abs/1902.00164)]
- [EQUATE] **EQUATE: A Benchmark Evaluation Framework for Quantitative Reasoning in Natural Language Inference**, CoNLL 2019 [[paper](https://arxiv.org/abs/1901.03735)]
- :fire: [NumerSense] **Birds have four legs?! NumerSense: Probing Numerical Commonsense Knowledge of Pre-trained Language Models**, EMNLP 2020 [[paper](https://arxiv.org/abs/2005.00683)]
- [MNS] **Machine Number Sense: A Dataset of Visual Arithmetic Problems for Abstract and Relational Reasoning**, AAAI 2020 [[paper](https://arxiv.org/abs/2004.12193)]
- [P3] **Programming Puzzles**, NeurIPS 2021 [[paper](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/3988c7f88ebcb58c6ce932b957b6f332-Abstract-round1.html)]
- [NOAHQA] **NOAHQA: Numerical Reasoning with Interpretable Graph Question Answering Dataset**,  Findings of EMNLP 2021 [[paper](https://arxiv.org/abs/2109.10604)]
- [ConvFinQA] **ConvFinQA: Exploring the Chain of Numerical Reasoning in Conversational Finance Question Answering**, arXiv:2210.03849 [[paper](https://arxiv.org/abs/2210.03849)]
- [PGDP5K] **PGDP5K: A Diagram Parsing Dataset for Plane Geometry Problems**, arXiv:2205.0994 [[paper](https://arxiv.org/abs/2205.09947)]
- [GeoRE] **GeoRE: A Relation Extraction Dataset for Chinese Geometry Problems**, NeurIPS 2021 MATHAI4ED Workshop [[paper](https://mathai4ed.github.io/papers/papers/paper_6.pdf)]
- :fire: [ScienceQA] **Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering**, NeurIPS 2022 [[paper](https://arxiv.org/abs/2209.09513)]



## 🧩 Neural Networks for Mathematical Reasoning

### General Neural Networks

- [LSTM] **Long short-term memory**, Neural computation 1997 [[paper](https://ieeexplore.ieee.org/abstract/document/6795963)]
- [Seq2Seq] **Sequence to sequence learning with neural networks**, NeurIPS 2014 [[paper](https://proceedings.neurips.cc/paper/2014/hash/a14ac55a4f27472c5d894ec1c3c743d2-Abstract.html)]
- [GRU] **Learning Phrase Representations using RNN Encoder--Decoder for Statistical Machine Translation**, EMNLP 2014 [[paper](https://arxiv.org/abs/1406.1078)]
- [Attention] **Neural machine translation by jointly learning to align and translate**, arXiv:1409.0473 [[paper](https://arxiv.org/abs/1409.0473)]
- [Attention] **Show, attend and tell: Neural image caption generation with visual attention**, ICML 2015 [[paper](https://arxiv.org/abs/1502.03044)]
- [Faster-RCNN] **Faster r-cnn: Towards real-time object detection with region proposal networks**, NeurIPS 2015 [[paper](https://arxiv.org/abs/1506.01497)]
- [TreeLSTM] **Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks**, ACL 2015 [[paper](https://arxiv.org/abs/1503.00075)]
- [BiLSTM] **Google's neural machine translation system: Bridging the gap between human and machine translation**, arXiv:1609.08144 [[paper](https://arxiv.org/abs/1609.08144)]
- [ResNet] **Deep residual learning for image recognition**, CVPR 2016 [[paper](https://arxiv.org/abs/1512.03385)]
- [ConvS2S] **Convolutional sequence to sequence learning**, ICML 2017 [[paper](https://arxiv.org/abs/1705.03122)]
- [Top-Down Attention] **Bottom-up and top-down attention for image captioning and visual question answering**, CVPR 2018 [[paper](https://arxiv.org/abs/1707.07998)]
- [FiLM] **Film: Visual reasoning with a general conditioning layer**, AAAI 2018 [[paper](https://arxiv.org/abs/1709.07871)]
- [BAN] **Bilinear Attention Networks**, NeurIPS 2018 [[paper](https://arxiv.org/abs/1805.07932)]
- [DAFA] **Dynamic Fusion With Intra-and Inter-Modality Attention Flow for Visual Question Answering**, CVPR 2018 [[paper](https://arxiv.org/abs/1812.05252)]

### Seq2Seq Networks for Math

- :fire: [DNS] **Deep Neural Solver for Math Word Problems**, EMNLP 2017 [[paper](https://aclanthology.org/D17-1088/)]
- :fire: [AnsRat] **Program induction by rationale generation: Learning to solve and explain algebraic word problems**, ACL 2017 [[paper](https://arxiv.org/abs/1705.04146)]
- [Math-EN] **Translating a Math Word Problem to a Expression Tree**, EMNLP 2018 [[paper](https://arxiv.org/abs/1811.05632)]
- [CASS] **Neural math word problem solver with reinforcement learning**, COLING 2018 [[paper](https://aclanthology.org/C18-1018/)]
- [SelfAtt] **Data-driven methods for solving algebra word problems**, arXiv:1804.10718 [[paper](https://arxiv.org/abs/1804.10718)]
- [S-Aligned] **Semantically-Aligned Equation Generation for Solving and Reasoning Math Word Problems**, NAACL 2019 [[paper](https://aclanthology.org/N19-1272/)]
- [T-RNN] **Template-based math word problem solvers with recursive neural networks**, AAAI 2019 [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/4697)]
- [GROUP-ATT] **Modeling intra-relation in math word problems with different functional multi-head attentions**, ACL 2019 [[paper](https://aclanthology.org/P19-1619/)]
- [QuaSP+] **QUAREL: A Dataset and Models for Answering Questions about Qualitative Relationships**, AAAI 2019 [[paper](https://arxiv.org/abs/1811.08048)]
- [SMART] **SMART: A Situation Model for Algebra Story Problems via Attributed Grammar**, AAAI 2021 [[paper](https://arxiv.org/abs/2012.14011)]

### Graph-based Networks for Math

- [AST-Dec] **Tree-structured decoding for solving math word problems**, EMNLP 2019 [[paper](https://aclanthology.org/D19-1241/)]
- :fire: [GTS] **A Goal-Driven Tree-Structured Neural Model for Math Word Problems**, IJCAI 2019 [[paper](https://www.ijcai.org/proceedings/2019/736)]
- [CoqGym] **Learning to Prove Theorems via Interacting with Proof Assistants**, ICML 2019 [[paper](https://arxiv.org/abs/1905.09381)]
- [KA-S2T] **A knowledge-aware sequence-to-tree network for math word problem solving**, EMNLP 2020 [[paper](https://aclanthology.org/2020.emnlp-main.579/)]
- [TSN-MD, NT-LSTM] **Solving arithmetic word problems by scoring equations with recursive neural networks**, Expert Systems with Applications 2021 [[paper](https://arxiv.org/abs/2009.05639)]
- [NS-Solver] **Neural-Symbolic Solver for Math Word Problems with Auxiliary Tasks**, ACL 2021 [[paper](https://arxiv.org/abs/2107.01431)]
- [NumS2T] **Math word problem solving with explicit numerical values**, ACL 2021 [[paper](https://aclanthology.org/2021.acl-long.455/)]
- [HMS] **Hms: A hierarchical solver with dependency-enhanced understanding for math word problem**, AAAI 2021 [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/16547)]
- [LBF] **Learning by fixing: Solving math word problems with weak supervision**, AAAI 2021 [[paper](https://arxiv.org/abs/2012.10582)]
- [Seq2DAG] **A bottom-up dag structure extraction model for math word problems**, AAAI 2021 [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/16075)]
- [Graph2Tree] **Graph-to-Tree Neural Networks for Learning Structured Input-Output Translation with Applications to Semantic Parsing and Math Word Problem**, EMNLP 2020 [[paper](https://arxiv.org/abs/2004.13781)]
- [Multi-E/D] **Solving math word problems with multi-encoders and multi-decoders**, COLING 2020 [[paper](https://aclanthology.org/2020.coling-main.262/)]
- :fire: [Graph2Tree] **Graph-to-Tree Learning for Solving Math Word Problems**, ACL 2020 [[paper](https://aclanthology.org/2020.acl-main.362/)]
- [EEH-G2T] **An edge-enhanced hierarchical graph-to-tree network for math word problem solving**, EMNLP 2021 [[paper](https://aclanthology.org/2021.findings-emnlp.127/)]

### Other Neural Networks for Math

- [DeepMath] **Deepmath-deep sequence models for premise selection**, NeurIPS 2016 [[paper](https://arxiv.org/abs/1606.04442)]
- [Holophrasm] **Holophrasm: a neural automated theorem prover for higher-order logic**, arXiv:1608.02644 [[paper](https://arxiv.org/abs/1608.02644)]
- :fire: [CNNTP, WaveNetTP] **Deep network guided proof search**, arXiv:1701.06972 [[paper](https://arxiv.org/abs/1701.06972)]
- :fire: [MathDQN] **Mathdqn: Solving arithmetic word problems via deep reinforcement learning**, AAAI 2018 [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/11981)]
- [DDT] **Solving math word problems with double-decoder transformer**, arXiv:1908.10924 [[paper](https://arxiv.org/abs/1908.10924)]
- [DeepHOL] **HOList: An environment for machine learning of higher order logic theorem proving**, ICML 2019 [[paper](https://arxiv.org/abs/1904.03241)]
- [NGS] **GeoQA: A Geometric Question Answering Benchmark Towards Multimodal Numerical Reasoning**, Findings of ACL 2021 [[paper](https://aclanthology.org/2021.findings-acl.46.pdf)]
- [PGDPNet] **Learning to Understand Plane Geometry Diagram**, NeurIPS 2022 MATH-AI Workshop [[paper](https://mathai2022.github.io/papers/6.pdf)]



## 📜 Pre-trained Language Models for Mathematical Reasoning

### General Pre-trained Language Models (<100B)

- [Transformer] **Attention is all you need**, NeurIPS 2017 [[paper](https://arxiv.org/abs/1706.03762)]
- [BERT] **Bert: Pre-training of deep bidirectional transformers for language understanding**, arXiv:1810.04805 [[paper](https://arxiv.org/abs/1810.04805)]
- [T5] **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer**, JMLR 2020 [[paper](https://arxiv.org/abs/1910.10683)]
- [RoBERTa] **Roberta: A robustly optimized bert pretraining approach**, arXiv:1907.11692 [[paper](https://arxiv.org/abs/1907.11692)]
- [GPT-2, 1.5B] **Language models are unsupervised multitask learners**, OpenAI Blog, 2020 [[paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)]
- [BART] **BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension**, ACL 2020 [[paper](https://arxiv.org/abs/1910.13461)]
- [ALBERT] **Albert: A lite bert for self-supervised learning of language representations**, arXiv:1909.11942 [[paper](https://arxiv.org/abs/1909.11942)]
- [GPT-Neo] **The pile: An 800gb dataset of diverse text for language modeling**, arXiv:2101.00027 [[paper](https://arxiv.org/abs/2101.00027)]
- [VL-T5] **Unifying Vision-and-Language Tasks via Text Generation**, ICML 2021 [[paper](https://arxiv.org/abs/2102.02779)]

### Self-Supervised Learning for Math

- :fire: [GenBERT] **Injecting numerical reasoning skills into language models**, ACL 2020 [[paper](https://arxiv.org/abs/2004.04487)]
- :fire: [GPT-f] **Generative language modeling for automated theorem proving**, arXiv:2009.03393 [[paper](https://arxiv.org/abs/2009.03393)]
- [LISA] **LISA: Language models of ISAbelle proofs**, AITP 2021 [[paper](http://aitp-conference.org/2021/abstract/paper_17.pdf)]
- [MATH-PLM] **Measuring Mathematical Problem Solving With the MATH Dataset**, NeurIPS 2021 [[paper](https://arxiv.org/abs/2103.03874)]
- [LIME] **Lime: Learning inductive bias for primitives of mathematical reasoning**, ICML 2021 [[paper](https://arxiv.org/abs/2101.06223)]
- [NF-NSM] **Injecting Numerical Reasoning Skills into Knowledge Base Question Answering Models**, arXiv:2112.06109 [[paper](https://arxiv.org/abs/2112.06109)]
- [MWP-BERT] **MWP-BERT: Numeracy-augmented pre-training for math word problem solving**, Findings of NAACL 2022 [[paper](https://arxiv.org/abs/2107.13435)]
- [HTPS] **HyperTree Proof Search for Neural Theorem Proving**, arXiv:2205.11491 [[paper](https://arxiv.org/abs/2205.11491)]
- [Thor] **Thor: Wielding Hammers to Integrate Language Models and Automated Theorem Provers**, arXiv:2205.10893 [[paper](https://arxiv.org/abs/2205.10893)]
- [Set] **Insights into pre-training via simpler synthetic tasks**, arXiv:2206.10139 [[paper](https://arxiv.org/abs/2206.10139)]
- [PACT] **Proof artifact co-training for theorem proving with language models**, ICLR 2022 [[paper](https://arxiv.org/abs/2102.06203)]
- :fire: [TaPEX] **TAPEX: Table Pre-training via Learning a Neural SQL Executor**, ICLR 2022 [[paper](https://arxiv.org/abs/2107.07653)]
- :fire: [Minerva] **Solving quantitative reasoning problems with language models**, NeurIPS 2022 [[paper](https://arxiv.org/abs/2206.14858)]

### Task-specific Fine-tuning for Math

- [EPT] **Point to the expression: Solving algebraic word problems using the expression-pointer transformer model**, EMNLP 2020 [[paper](https://aclanthology.org/2020.emnlp-main.308/)]
- [Generate \& Rank] **Generate \& Rank: A Multi-task Framework for Math Word Problems**, EMNLP 2021 [[paper](https://arxiv.org/abs/2109.03034)]
- [RPKHS] **Improving Math Word Problems with Pre-trained Knowledge and Hierarchical Reasoning**, EMNLP 2021 [[paper](https://aclanthology.org/2021.emnlp-main.272/)]
- [PatchTRM] **IconQA: A New Benchmark for Abstract Diagram Understanding and Visual Language Reasoning**, NeurIPS 2021 [[paper](https://arxiv.org/abs/2110.13214)]
- :fire: [GSM8K-PLM] **Training verifiers to solve math word problems**, arXiv:2110.14168 [[paper](https://arxiv.org/abs/2110.14168)]
- :fire: [Inter-GPS] **Inter-GPS: Interpretable Geometry Problem Solving with Formal Language and Symbolic Reasoning**, ACL 2021 [[paper](https://aclanthology.org/2021.acl-long.528/)]
- [Aristo] From ‘F’to ‘A’on the NY regents science exams: An overview of the aristo project, AI Magazine 2020 [paper]
- [FinQANet] **FinQA: A Dataset of Numerical Reasoning over Financial Data**, EMNLP 2021 [[paper](https://arxiv.org/abs/2109.00122)]
- [TAGOP] **TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance**, ACL-JCNLP 2021 [[paper](https://arxiv.org/abs/2105.07624)]
- [LAMT] **Linear algebra with transformers**, arXiv:2112.01898 [[paper](https://arxiv.org/abs/2112.01898)]
- :fire: [Scratchpad] **Show your work: Scratchpads for intermediate computation with language models**, arXiv:2112.00114 [[paper](https://arxiv.org/abs/2112.00114)]
- [Self-Sampling] **Learning from Self-Sampled Correct and Partially-Correct Programs**, arXiv:2205.14318 [[paper](https://arxiv.org/abs/2205.14318)]
- [DeductReasoner] **Learning to Reason Deductively: Math Word Problem Solving as Complex Relation Extraction**, ACL 2022 [[paper](https://arxiv.org/abs/2203.10316)]
- [DPE-NGS] **An Augmented Benchmark Dataset for Geometric Question Answering through Dual Parallel Text Encoding**, COLING 2022 [[paper](https://aclanthology.org/2022.coling-1.130/)]
- [BERT-TD+CL] **Seeking Patterns, Not just Memorizing Procedures: Contrastive Learning for Solving Math Word Problems**, Findings of ACL 2022 [[paper](https://arxiv.org/abs/2110.08464)]
- [MT2Net] **MultiHiertt: Numerical Reasoning over Multi Hierarchical Tabular and Textual Data**, ACL 2022 [[paper](https://aclanthology.org/2022.acl-long.454/)]
- [miniF2F-PLM] **MiniF2F: a cross-system benchmark for formal Olympiad-level mathematics**, ICLR 2022 [[paper](https://arxiv.org/abs/2109.00110)]
- :fire: [NaturalProver] **NaturalProver: Grounded Mathematical Proof Generation with Language Models**, NeurIPS 2022 [[paper](https://arxiv.org/abs/2205.12910)]
- :fire: [UniGeo] **UniGeo: Unifying Geometry Logical Reasoning via Reformulating Mathematical Expression**, EMNLP 2022 [[paper](https://lupantech.github.io/papers/emnlp22_unigeo.pdf)]
- :fire: [Bhaskara] **Lila: A Unified Benchmark for Mathematical Reasoning**, EMNLP 2022 [[paper](https://arxiv.org/abs/2210.17517)]



## 🌠 In-context Learning for Mathematical Reasoning

### General Large Language Models (100B+)

- :fire: [GPT-3, 175B] **Language models are few-shot learners**, NeurIPS 2020 [[paper](https://arxiv.org/abs/2005.14165)]
- :fire: [Codex, 175B] **Evaluating large language models trained on code**, arXiv:2107.03374 [[paper](https://arxiv.org/abs/2107.03374)]
- :fire: [PaLM, 540B] **PaLM: Scaling Language Modeling with Pathways**, arXiv:2204.02311 [[paper](https://arxiv.org/abs/2204.02311)]
- :fire: [ChatGPT, 175B] **ChatGPT: Optimizing Language Models for Dialogue**, November 30, 2022 [[website](https://openai.com/blog/chatgpt/)]
- :question: [GPT-4]

### In-context Example Selection

- :fire: [Few-shot-CoT] **Chain of thought prompting elicits reasoning in large language models**, NeurIPS 2022 [[paper](https://arxiv.org/abs/2201.11903)]
- [Retrieval] **Learning to retrieve prompts for in-context learning**, NAACL-HLT 2022 [[paper](https://arxiv.org/abs/2112.08633)]
- :fire: [PromptPG-CoT] **Dynamic Prompt Learning via Policy Gradient for Semi-structured Mathematical Reasoning**, arXiv:2209.14610 [[paper](https://arxiv.org/abs/2209.14610)]
- [Retrieval-CoT] **Automatic Chain of Thought Prompting in Large Language Models**, arXiv:2210.03493 [[paper](https://arxiv.org/abs/2210.03493)]
- [Generate] **Generate rather than retrieve: Large language models are strong context generators**, arXiv:2209.10063 [[paper](https://arxiv.org/abs/2209.10063)]
- [Complexity-CoT] **Complexity-Based Prompting for Multi-Step Reasoning,** arXiv:2210.00720 [[paper](https://arxiv.org/abs/2210.00720)]
- [Auto-CoT] **Automatic Chain of Thought Prompting in Large Language Models**, arXiv:2210.03493 [[paper](https://arxiv.org/abs/2210.03493)]

### High-quality Reasoning Chains

- :fire: [Self-Consistency-CoT] **Self-consistency improves chain of thought reasoning in language models**, arXiv:2203.11171 [[paper](https://arxiv.org/abs/2203.11171)]
- :fire: [Least-to-most CoT] **Least-to-Most Prompting Enables Complex Reasoning in Large Language Models**, arXiv:2205.10625 [[paper](https://arxiv.org/abs/2205.10625)]
- **On the Advance of Making Language Models Better Reasoners**, arXiv:2206.02336 [[paper](https://arxiv.org/abs/2206.02336)]
- **Decomposed prompting: A modular approach for solving complex tasks**, arXiv:2210.02406 [[paper](https://arxiv.org/abs/2210.02406)]
- **PAL: Program-aided Language Models**, arXiv:2211.10435 [[paper](https://arxiv.org/abs/2211.10435)]
- :fire: [Few-shot-PoT] **Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks**, arXiv:2211.12588 [[paper](https://arxiv.org/abs/2211.12588)]



## ♣️ Other Related Work for Mathematical Reasoning

### Early Work

- **Empirical explorations of the geometry theorem machine**, Western Joint IRE-AIEE-ACM Computer Conference 1960 [[paper](https://dl.acm.org/doi/10.1145/1460361.1460381)]
- **Basic principles of mechanical theorem proving in elementary geometries**, Journal of Automated Reasoning 1986 [[paper](https://link.springer.com/article/10.1007/BF02328447)]
- **Automated generation of readable proofs with geometric invariants**, Journal of Automated Reasoning 1996 [[paper](https://link.springer.com/article/10.1007/BF00283133)]

### Datasets

- :fire: [TextbookQA] **Are You Smarter Than A Sixth Grader? Textbook Question Answering for Multimodal Machine Comprehension**, CVPR 2017 [[paper](https://ieeexplore.ieee.org/document/8100054)]
- :fire: [Raven] **Raven: A dataset for relational and analogical visual reasoning**, CVPR 2019 [[paper](https://arxiv.org/abs/1903.02741)]
- [APPS] **Measuring Coding Challenge Competence With APPS**, NeurIPS 2021 [[paper](https://arxiv.org/abs/2105.09938)]
- [PhysNLU] **PhysNLU: A Language Resource for Evaluating Natural Language Understanding and Explanation Coherence in Physics**, 2022 [[paper](https://arxiv.org/abs/2201.04275)]

### Methods

- **My computer is an honor student—but how intelligent is it? Standardized tests as a measure of AI**, AI Magazine 2016 [[paper](https://ojs.aaai.org//index.php/aimagazine/article/view/2636)]
- **Learning pipelines with limited data and domain knowledge: A study in parsing physics problems**, NeurIPS 2018 [[paper](https://proceedings.neurips.cc/paper/2018/hash/ac627ab1ccbdb62ec96e702f07f6425b-Abstract.html)]
- **Automatically proving plane geometry theorems stated by text and diagram**, International Journal of Pattern Recognition and Artificial Intelligence 2019 [[paper](https://www.worldscientific.com/doi/abs/10.1142/S0218001419400032)]
- **Classification and Clustering of arXiv Documents, Sections, and Abstracts, Comparing Encodings of Natural and Mathematical Language**, JCDL 2020 [[paper](https://arxiv.org/abs/2005.11021)]

### Latest Work (To be classified)

- :fire: **Advancing mathematics by guiding human intuition with AI**, Nature 2021 [[paper](https://www.nature.com/articles/s41586-021-04086-x)]
- [MWPToolkit] **Mwptoolkit: an open-source framework for deep learning-based math word problem solvers**, AAAI 2022 [[paper](https://arxiv.org/abs/2109.00799)]
- **A deep reinforcement learning agent for geometry online tutoring**, Knowledge and Information Systems 2022 [[paper](https://link.springer.com/article/10.1007/s10115-022-01804-3)]
- **ELASTIC: Numerical Reasoning with Adaptive Symbolic Compiler**, NeurIPS 2022 [[paper](https://arxiv.org/abs/2210.10105)]
- **Solving math word problems with process and outcome-based feedback**, arXiv:2211.14275 [[paper](https://arxiv.org/abs/2211.14275)]
- **APOLLO: An Optimized Training Approach for Long-form Numerical Reasoning**, arXiv:2212.07249 [[paper](https://arxiv.org/abs/2212.07249)]
- **Enhancing Financial Table and Text Question Answering with Tabular Graph and Numerical Reasoning**, AACL 2022 [[paper](https://aclanthology.org/2022.aacl-main.72/)]
- **DyRRen: A Dynamic Retriever-Reranker-Generator Model for Numerical Reasoning over Tabular and Textual Data**, AAAI 2023 [[paper](https://arxiv.org/abs/2211.12668)]
- **Generalizing Math Word Problem Solvers via Solution Diversification**, arXiv:2212.00833 [[paper](https://arxiv.org/abs/2212.00833)]
- **Textual Enhanced Contrastive Learning for Solving Math Word Problems**, arXiv:2211.16022 [[paper](https://arxiv.org/abs/2211.16022)]
- **Analogical Math Word Problems Solving with Enhanced Problem-Solution Association**, EMNLP 2022 [[paper](https://arxiv.org/abs/2212.00837)]



## Citation

If you find this repo useful, please kindly cite our survey:

```
@article{lu2022dl4math,
  title={A Survey of Deep Learning for Mathematical Reasoning},
  author={Lu, Pan and Qiu, Liang and Yu, Wenhao and Welleck, Sean and Chang, Kai-Wei},
  journal={arXiv preprint arXiv:2212.10535},
  year={2022}
}
```

