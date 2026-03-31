import type { TimelineItem } from "../components/features/timeline/types";

export const timelineData: TimelineItem[] = [
	{
		id: "sbi-research",
		title: "SBI Neural Posterior Estimation 研究",
		description:
			"EPSRC 资助的本科生研究项目。构建了 3 层 DNN 用于 Simulation-Based Inference，集成 normalizing flows 和 CLT 结构正则化，并在诺丁汉大学本科生研究海报展上展示成果。",
		type: "project",
		startDate: "2025-06-01",
		endDate: "2025-08-31",
		location: "Nottingham, UK",
		organization: "University of Nottingham (EPSRC)",
		skills: ["Python", "PyTorch", "Normalizing Flows", "SBI", "DNN"],
		achievements: [
			"构建 normalizing flow 条件架构用于 Poisson 模型推断",
			"证明 CLT 正则化提高了推断稳定性",
			"在诺丁汉大学本科生研究海报展上展示",
		],
		links: [
			{
				name: "GitHub",
				url: "https://github.com/Yingurt001/SBI-summer-project",
				type: "project",
			},
		],
		icon: "material-symbols:science",
		color: "#059669",
		featured: true,
	},
	{
		id: "mcm-2025",
		title: "MCM 2025 — Outstanding Prize",
		description:
			"队长。基于 NSGA-II 优化朱诺旅游可持续发展，构建 SPEM 模型量化环境-经济权衡，推广至巴厘岛和圣芭芭拉。",
		type: "achievement",
		startDate: "2025-01-15",
		endDate: "2025-02-15",
		location: "Massachusetts, US",
		organization: "COMAP MCM",
		skills: ["NSGA-II", "AHP", "CRITIC", "Dynamic Programming", "Python"],
		achievements: [
			"Outstanding Prize（特等奖）",
			"多目标优化 15 条消费路径",
			"模型推广至巴厘岛和圣芭芭拉",
		],
		links: [
			{
				name: "GitHub",
				url: "https://github.com/Yingurt001/Sustainable-Tourism-Optimization-in-Juneau-via-NSGA-",
				type: "project",
			},
		],
		icon: "material-symbols:emoji-events",
		color: "#FFD700",
		featured: true,
	},
	{
		id: "credit-risk-kan",
		title: "KAN-GRU/LSTM 贷款违约预测",
		description:
			"开发 GRU-KAN 和 LSTM-KAN 架构用于贷款违约早期检测（提前 3-8 个月），AUC 较 GRU baseline 提升 12%。论文投稿 Applied Soft Computing，已被 CSCR 2024 接收。",
		type: "project",
		startDate: "2025-01-01",
		endDate: "2025-03-31",
		location: "Nottingham, UK",
		organization: "University of Nottingham",
		skills: ["PyTorch", "KAN", "GRU", "LSTM", "Freddie Mac Data"],
		achievements: [
			"AUC 较 GRU baseline 提升 12%",
			"论文投稿 Applied Soft Computing",
			"CSCR 2024 会议接收",
		],
		links: [
			{
				name: "GitHub",
				url: "https://github.com/Yingurt001/Imbalancing-antifraud",
				type: "project",
			},
			{
				name: "Paper",
				url: "https://arxiv.org/abs/2507.13685",
				type: "website",
			},
		],
		icon: "material-symbols:code",
		color: "#7C3AED",
		featured: true,
	},
	{
		id: "mcm-2024",
		title: "MCM 2024 — Outstanding Prize",
		description:
			"队长。用 Markov Chain + LSTM 预测网球比赛中的 momentum，通过 MCMC 模拟和 K-S 检验验证动量非随机性（p < 0.0001），交叉验证准确率 76.3%。",
		type: "achievement",
		startDate: "2024-01-15",
		endDate: "2024-02-15",
		location: "Massachusetts, US",
		organization: "COMAP MCM",
		skills: ["Markov Chain", "LSTM", "MCMC", "EMA", "Python"],
		achievements: [
			"Outstanding Prize（特等奖）",
			"动量分类交叉验证准确率 76.3%",
			"K-S 检验 p < 0.0001",
		],
		icon: "material-symbols:emoji-events",
		color: "#FFD700",
		featured: true,
	},
	{
		id: "t-bilstm",
		title: "T-BiLSTM 贷后违约预测",
		description:
			"开发 Transformer-BiLSTM 架构，处理 2000 万+ 贷款记录的借款人还款轨迹与宏观经济信号。假阴性较 LSTM/GRU baseline 降低 15%。论文投稿 Information System Frontiers。",
		type: "project",
		startDate: "2024-03-01",
		endDate: "2024-08-31",
		location: "Ningbo, CN",
		organization: "University of Nottingham Ningbo China",
		skills: ["Transformer", "BiLSTM", "Python", "Feature Engineering"],
		achievements: [
			"假阴性较 baseline 降低 15%",
			"处理 2000 万+ 贷款记录",
			"论文投稿 Information System Frontiers",
		],
		links: [
			{
				name: "GitHub",
				url: "https://github.com/Yingurt001/Transformer-Enhanced-BiLSTM-for-Post-Loan-Default-Prediction-",
				type: "project",
			},
			{
				name: "Paper",
				url: "https://arxiv.org/abs/2508.00415",
				type: "website",
			},
		],
		icon: "material-symbols:code",
		color: "#7C3AED",
	},
	{
		id: "shuwei-2024",
		title: "数维杯数模挑战赛 — Outstanding Prize",
		description:
			"队长。使用逻辑回归和灰色预测算法建模脱硫灰催化反应，预测最优催化剂混合比，预测损失表现获特等奖。",
		type: "achievement",
		startDate: "2023-12-15",
		endDate: "2024-01-15",
		location: "Inner Mongolia, CN",
		organization: "数维杯",
		skills: ["Logistic Regression", "Grey Prediction", "Python"],
		achievements: [
			"Outstanding Prize（特等奖）",
			"催化剂混合比预测最优",
		],
		links: [
			{
				name: "GitHub",
				url: "https://github.com/Yingurt001/Pyrolysis-Process-Prediction-ML",
				type: "project",
			},
		],
		icon: "material-symbols:emoji-events",
		color: "#FFD700",
	},
	{
		id: "vegatable-pricing",
		title: "蔬菜自动定价与补货策略",
		description:
			"国赛数模。用 Prophet + 模拟退火 + 遗传算法，构建七天动态定价与补货方案，最小化损耗和缺货。",
		type: "project",
		startDate: "2023-09-01",
		endDate: "2023-10-31",
		location: "Beijing, CN",
		organization: "全国大学生数学建模竞赛",
		skills: ["Prophet", "Simulated Annealing", "Genetic Algorithm", "MIQP"],
		achievements: [
			"National 3rd Prize（国三）",
			"七天动态定价与补货优化",
		],
		links: [
			{
				name: "GitHub",
				url: "https://github.com/Yingurt001/vegetable-pricing-replenishment-model",
				type: "project",
			},
		],
		icon: "material-symbols:code",
		color: "#EA580C",
	},
	{
		id: "mcm-2023",
		title: "数学建模竞赛 — National 1st Prize",
		description:
			"2023 年数学建模竞赛全国一等奖。",
		type: "achievement",
		startDate: "2023-04-01",
		organization: "数学建模竞赛",
		achievements: [
			"National 1st Prize（全国一等奖）",
		],
		icon: "material-symbols:emoji-events",
		color: "#DC2626",
	},
	{
		id: "nottingham-bsc",
		title: "BSc Statistics — University of Nottingham",
		description:
			"诺丁汉大学数学科学学院统计学本科，GPA 3.95，年级前 5%。主修复分析、微分方程、向量微积分、实分析、科学计算、概率统计理论等。",
		type: "education",
		startDate: "2022-09-01",
		endDate: "2026-05-31",
		location: "Nottingham, UK",
		organization: "University of Nottingham",
		skills: ["Statistics", "Mathematics", "Python", "R", "MATLAB"],
		achievements: [
			"GPA 3.95, Top 5%",
			"EPSRC 资助本科生研究",
			"多项数模竞赛获奖",
		],
		icon: "material-symbols:school",
		color: "#2563EB",
		featured: true,
	},
	{
		id: "blog-launch",
		title: "个人博客上线",
		description:
			"基于 Astro + Mizuki 主题搭建个人博客，部署于 Vercel。记录生活、研究和技术分享。",
		type: "project",
		startDate: "2026-03-30",
		skills: ["Astro", "TypeScript", "Vercel", "Markdown"],
		achievements: [
			"从零搭建并部署上线",
			"迁移旧网站 16 篇技术文章",
		],
		links: [
			{
				name: "博客",
				url: "https://andrea-blog-gilt.vercel.app/",
				type: "website",
			},
		],
		icon: "material-symbols:edit-note",
		color: "#059669",
	},
];
