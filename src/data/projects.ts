// Project data configuration file
// Used to manage data for the project display page

export interface Project {
	id: string;
	title: string;
	description: string;
	image: string;
	category: "web" | "mobile" | "desktop" | "other";
	techStack: string[];
	status: "completed" | "in-progress" | "planned";
	liveDemo?: string;
	sourceCode?: string;
	visitUrl?: string;
	startDate: string;
	endDate?: string;
	featured?: boolean;
	tags?: string[];
	showImage?: boolean;
}

export const projectsData: Project[] = [
	{
		id: "sbi-npe",
		title: "SBI Neural Posterior Estimation",
		description:
			"EPSRC 资助研究。构建 3 层 DNN 集成 normalizing flows 与 CLT 正则化，用于 Simulation-Based Inference，在 gamma/beta 数据集上验证。",
		image: "/assets/projects/sbi-npe.jpg",
		category: "other",
		techStack: ["Python", "PyTorch", "Normalizing Flows", "SBI"],
		status: "completed",
		sourceCode: "https://github.com/Yingurt001/SBI-summer-project",
		startDate: "2025-06-01",
		endDate: "2025-08-31",
		featured: true,
		tags: ["Research", "ML", "Bayesian"],
	},
	{
		id: "kan-credit-risk",
		title: "KAN-GRU/LSTM 贷款违约预测",
		description:
			"用 Kolmogorov–Arnold Networks 改造 GRU 与 LSTM 做贷款违约早期预测，提前 3 个月准确率超过 92%、提前 8 个月超过 88%。合作项目，本人第三作者，论文被 Applied Soft Computing 2025 接收。",
		image: "/assets/projects/kan-credit.jpg",
		category: "other",
		techStack: ["Python", "PyTorch", "KAN", "GRU", "LSTM"],
		status: "completed",
		sourceCode: "https://github.com/Yingurt001/Imbalancing-antifraud",
		startDate: "2025-01-01",
		endDate: "2025-03-31",
		featured: true,
		tags: ["Research", "Credit Risk", "Deep Learning"],
	},
	{
		id: "t-bilstm",
		title: "ResE-BiLSTM 贷后违约检测",
		description:
			"时序驱动的 ResE-BiLSTM 框架，在 Freddie Mac 房贷数据集 44 个独立 cohort 上与五个基线对比。合作项目，本人第三作者，论文被 CSCR III 2024 接收。",
		image: "/assets/projects/t-bilstm.jpg",
		category: "other",
		techStack: ["Python", "PyTorch", "Transformer", "BiLSTM"],
		status: "completed",
		sourceCode:
			"https://github.com/Yingurt001/Transformer-Enhanced-BiLSTM-for-Post-Loan-Default-Prediction-",
		startDate: "2024-03-01",
		endDate: "2024-08-31",
		featured: true,
		tags: ["Research", "Credit Risk", "Time Series"],
	},
	{
		id: "hyperbolic-nn",
		title: "双曲神经网络研究",
		description:
			"双曲空间上的神经网络：在 Klein 模型上补全全连接、MLR 分类头、归一化等组件，与 Poincaré / Lorentz 实现做系统对比与数值稳定性实验。实验跑在 Nottingham Ada 与 CINECA Leonardo 集群上。",
		image: "/assets/projects/hyperbolic.jpg",
		category: "other",
		techStack: ["Python", "PyTorch", "HPC", "SLURM"],
		status: "in-progress",
		sourceCode: "https://github.com/Yingurt001/hyperbolic-nn-hpc",
		startDate: "2026-02-01",
		featured: true,
		tags: ["Research", "Hyperbolic Geometry", "Neural Networks"],
	},
	{
		id: "mcm-2025-juneau",
		title: "MCM 2025 — 朱诺旅游优化",
		description:
			"Honorable Mention。基于 NSGA-II 多目标优化朱诺旅游可持续发展，构建 SPEM 模型，推广至巴厘岛和圣芭芭拉。",
		image: "/assets/projects/mcm-juneau.jpg",
		category: "other",
		techStack: [
			"Python",
			"NSGA-II",
			"AHP",
			"CRITIC",
			"Dynamic Programming",
		],
		status: "completed",
		sourceCode:
			"https://github.com/Yingurt001/Sustainable-Tourism-Optimization-in-Juneau-via-NSGA-",
		startDate: "2025-01-15",
		endDate: "2025-02-15",
		tags: ["Math Modeling", "Optimization", "MCM"],
	},
	{
		id: "mcm-2024-tennis",
		title: "MCM 2024 — 网球动量预测",
		description:
			"Successful Participant。Markov Chain + LSTM 预测网球比赛 momentum，MCMC 模拟验证，交叉验证准确率 76.3%。",
		image: "/assets/projects/mcm-tennis.jpg",
		category: "other",
		techStack: ["Python", "LSTM", "Markov Chain", "MCMC"],
		status: "completed",
		sourceCode:
			"https://github.com/Yingurt001/Tennis-Match-Outcome-Prediction-via-Ensemble-Learning",
		startDate: "2024-01-15",
		endDate: "2024-02-15",
		tags: ["Math Modeling", "Sports Analytics", "MCM"],
	},
	{
		id: "apple-recognition",
		title: "苹果智能检测系统",
		description:
			"APMCM 2023 一等奖。基于 OpenCV + CNN/ResNet50 的果园苹果计数、定位、成熟度分类与品质评估。",
		image: "/assets/projects/apple.jpg",
		category: "other",
		techStack: ["Python", "OpenCV", "CNN", "ResNet50"],
		status: "completed",
		sourceCode:
			"https://github.com/Yingurt001/Intelligent-Apple-Recognition",
		startDate: "2023-11-01",
		tags: ["Computer Vision", "Deep Learning", "Math Modeling"],
	},
	{
		id: "vegatable-pricing",
		title: "蔬菜自动定价与补货策略",
		description:
			"国赛数模 National 3rd Prize。Prophet + 模拟退火 + 遗传算法，七天动态定价与补货优化。",
		image: "/assets/projects/vegetable.jpg",
		category: "other",
		techStack: [
			"Python",
			"Prophet",
			"Simulated Annealing",
			"Genetic Algorithm",
		],
		status: "completed",
		sourceCode:
			"https://github.com/Yingurt001/vegetable-pricing-replenishment-model",
		startDate: "2023-09-01",
		endDate: "2023-10-31",
		tags: ["Math Modeling", "Optimization"],
	},
	{
		id: "andrea-blog",
		title: "Andrea's Blog",
		description:
			"基于 Astro + Mizuki 主题的个人博客，部署于 Vercel。记录生活、研究和技术分享。",
		image: "/assets/projects/blog.jpg",
		category: "web",
		techStack: ["Astro", "TypeScript", "Tailwind CSS", "Vercel"],
		status: "completed",
		sourceCode: "https://github.com/Yingurt001/andrea-blog",
		visitUrl: "https://andrea-blog-gilt.vercel.app/",
		startDate: "2026-03-30",
		tags: ["Blog", "Web"],
	},
	{
		id: "vibelog",
		title: "Vibelog",
		description: "记录日常 vibe 的个人 App。",
		image: "/assets/projects/vibelog.jpg",
		category: "mobile",
		techStack: ["TypeScript"],
		status: "completed",
		sourceCode: "https://github.com/Yingurt001/Vibelog",
		startDate: "2026-01-01",
		endDate: "2026-01-29",
		tags: ["App", "Personal"],
	},
	{
		id: "multi-device-sync",
		title: "多平台同步控制系统",
		description:
			"一个终端控制四台设备 (MacBook/Mac Mini/Windows/HPC)，支持任务派发、广播命令和控制面板。",
		image: "/assets/projects/multi-device.jpg",
		category: "desktop",
		techStack: ["Shell", "SSH", "Tailscale", "tmux"],
		status: "completed",
		sourceCode: "https://github.com/Yingurt001/multi-device-sync",
		startDate: "2026-03-01",
		endDate: "2026-03-11",
		tags: ["DevOps", "Automation"],
	},
	{
		id: "pyrolysis-ml",
		title: "脱硫灰催化反应 ML 预测",
		description:
			"数维杯国家二等奖。逻辑回归 + 灰色预测算法建模催化反应，预测最优催化剂混合比。",
		image: "/assets/projects/pyrolysis.jpg",
		category: "other",
		techStack: [
			"MATLAB",
			"Python",
			"Logistic Regression",
			"Grey Prediction",
		],
		status: "completed",
		sourceCode:
			"https://github.com/Yingurt001/Pyrolysis-Process-Prediction-ML",
		startDate: "2023-12-15",
		endDate: "2024-01-15",
		tags: ["Math Modeling", "Chemistry"],
	},
];

// Get project statistics
export const getProjectStats = () => {
	const total = projectsData.length;
	const completed = projectsData.filter(
		(p) => p.status === "completed",
	).length;
	const inProgress = projectsData.filter(
		(p) => p.status === "in-progress",
	).length;
	const planned = projectsData.filter((p) => p.status === "planned").length;

	return {
		total,
		byStatus: {
			completed,
			inProgress,
			planned,
		},
	};
};

// Get projects by category
export const getProjectsByCategory = (category?: string) => {
	if (!category || category === "all") {
		return projectsData;
	}
	return projectsData.filter((p) => p.category === category);
};

// Get featured projects
export const getFeaturedProjects = () => {
	return projectsData.filter((p) => p.featured);
};

// Get all tech stacks
export const getAllTechStack = () => {
	const techSet = new Set<string>();
	projectsData.forEach((project) => {
		project.techStack.forEach((tech) => {
			techSet.add(tech);
		});
	});
	return Array.from(techSet).sort();
};
