// Skill data configuration file
// Used to manage data for the skill display page

export interface Skill {
	id: string;
	name: string;
	description: string;
	icon: string; // Iconify icon name
	category: "frontend" | "backend" | "database" | "tools" | "other";
	level: "beginner" | "intermediate" | "advanced" | "expert";
	experience: {
		years: number;
		months: number;
	};
	projects?: string[]; // Related project IDs
	certifications?: string[];
	color?: string; // Skill card theme color
}

export const skillsData: Skill[] = [
	// 分类键沿用主题的 frontend/backend/database/tools/other，显示名在 i18n 里改成了：
	// frontend=机器学习与统计 · backend=编程语言与框架 · database=数学与理论 · tools=工具与算力 · other=研究之外
	// 年限按真正开始用的时间算，不按「听说过」算。

	// ── 机器学习与统计 ──
	{
		id: "pytorch",
		name: "PyTorch",
		description:
			"日常主力。双曲/流形上的层自己写，要兼顾 autograd 和数值稳定；多卡、混合精度、array job 都跑过。",
		icon: "simple-icons:pytorch",
		category: "frontend",
		level: "advanced",
		experience: { years: 2, months: 6 },
		projects: ["hyperbolic-nn", "t-bilstm", "kan-credit-risk"],
		color: "#EE4C2C",
	},
	{
		id: "hyperbolic-dl",
		name: "双曲与流形深度学习",
		description:
			"Poincaré / Lorentz / Klein 三种模型上的全连接、MLR、归一化组件与稳定性对比；正在把归一化推广到一般齐次空间。",
		icon: "material-symbols:blur-circular",
		category: "frontend",
		level: "advanced",
		experience: { years: 1, months: 0 },
		projects: ["hyperbolic-nn"],
		color: "#7C3AED",
	},
	{
		id: "bayesian-sbi",
		name: "贝叶斯推断与 SBI",
		description:
			"Simulation-Based Inference：用 normalizing flows 做 neural posterior estimation，从 EPSRC 暑研起步，一路带到扩散模型的后验采样。",
		icon: "material-symbols:query-stats",
		category: "frontend",
		level: "advanced",
		experience: { years: 1, months: 6 },
		projects: ["sbi-npe"],
		color: "#059669",
	},
	{
		id: "deep-learning",
		name: "深度学习",
		description:
			"Transformer、LSTM/GRU、KAN、JEPA、扩散与 flow matching。从复现论文到改架构、跑消融。",
		icon: "material-symbols:neurology",
		category: "frontend",
		level: "advanced",
		experience: { years: 2, months: 6 },
		projects: ["t-bilstm", "kan-credit-risk"],
		color: "#DB2777",
	},
	{
		id: "time-series",
		name: "时间序列",
		description:
			"贷款违约的提前预测、多变量异常检测、提前预警 benchmark 的协议设计与审计。",
		icon: "material-symbols:show-chart",
		category: "frontend",
		level: "intermediate",
		experience: { years: 2, months: 0 },
		projects: ["t-bilstm", "kan-credit-risk"],
		color: "#0EA5E9",
	},
	{
		id: "diffusion",
		name: "扩散模型与 Flow Matching",
		description:
			"科学逆问题上的后验采样：amortized flow 与 per-observation guidance 的对比与融合。浙大暑期研究。",
		icon: "material-symbols:blur-on",
		category: "frontend",
		level: "intermediate",
		experience: { years: 0, months: 4 },
		color: "#F59E0B",
	},

	// ── 编程语言与框架 ──
	{
		id: "python",
		name: "Python",
		description:
			"科研主语言。NumPy / pandas / scikit-learn，实验、脚本、自动化工具都用它写。大一开始，一直没停。",
		icon: "simple-icons:python",
		category: "backend",
		level: "expert",
		experience: { years: 4, months: 0 },
		color: "#3776AB",
	},
	{
		id: "r-lang",
		name: "R",
		description:
			"统计课程和作业里的建模与可视化，ggplot2 画图比 matplotlib 顺手。",
		icon: "simple-icons:r",
		category: "backend",
		level: "intermediate",
		experience: { years: 3, months: 0 },
		color: "#276DC3",
	},
	{
		id: "matlab",
		name: "MATLAB",
		description: "数模竞赛时期的主力：灰色预测、优化求解、快速原型。",
		icon: "material-symbols:function",
		category: "backend",
		level: "intermediate",
		experience: { years: 2, months: 6 },
		projects: ["pyrolysis-ml", "vegatable-pricing"],
		color: "#E16737",
	},
	{
		id: "astro-svelte",
		name: "Astro + Svelte + TypeScript",
		description:
			"这个博客本身：Astro 静态站、Svelte 5 交互组件、Tailwind。专注统计仪表盘就是拿它们搭的。",
		icon: "simple-icons:astro",
		category: "backend",
		level: "intermediate",
		experience: { years: 0, months: 6 },
		projects: ["andrea-blog"],
		color: "#FF5D01",
	},
	{
		id: "bash",
		name: "Shell / Bash",
		description: "集群提交脚本、launchd 定时任务、多设备调度，胶水语言。",
		icon: "simple-icons:gnubash",
		category: "backend",
		level: "advanced",
		experience: { years: 3, months: 0 },
		projects: ["multi-device-sync"],
		color: "#4EAA25",
	},

	// ── 数学与理论 ──
	{
		id: "statistics",
		name: "统计学",
		description:
			"概率论、统计推断、回归、贝叶斯方法。诺丁汉统计学本科，GPA 3.95。",
		icon: "material-symbols:bar-chart",
		category: "database",
		level: "expert",
		experience: { years: 4, months: 0 },
		color: "#2563EB",
	},
	{
		id: "optimization",
		name: "最优化",
		description:
			"NSGA-II、模拟退火、遗传算法、MIQP、动态规划。三次数模特等奖的工具箱。",
		icon: "material-symbols:trending-up",
		category: "database",
		level: "advanced",
		experience: { years: 3, months: 0 },
		projects: ["mcm-2025-juneau", "mcm-2024-tennis", "vegatable-pricing"],
		color: "#D97706",
	},
	{
		id: "analysis",
		name: "分析与线性代数",
		description:
			"实分析、复分析、向量微积分、微分方程、科学计算。本科的主菜。",
		icon: "material-symbols:calculate",
		category: "database",
		level: "advanced",
		experience: { years: 4, months: 0 },
		color: "#4F46E5",
	},
	{
		id: "differential-geometry",
		name: "微分几何",
		description:
			"流形、测地线、平行移动、齐次空间。为双曲网络和流形上的归一化打底，边用边补。",
		icon: "material-symbols:architecture",
		category: "database",
		level: "intermediate",
		experience: { years: 1, months: 0 },
		color: "#0D9488",
	},

	// ── 工具与算力 ──
	{
		id: "hpc",
		name: "HPC / Slurm",
		description:
			"Nottingham Ada、CINECA Leonardo、实验室 8×4090：作业调度、array job、离线环境、QoS 与拨款管理。",
		icon: "material-symbols:memory",
		category: "tools",
		level: "advanced",
		experience: { years: 1, months: 0 },
		projects: ["hyperbolic-nn", "multi-device-sync"],
		color: "#6366F1",
	},
	{
		id: "git",
		name: "Git / GitHub",
		description:
			"分支、rebase、pre-push hook，匿名投稿仓库的搭法也踩过一遍坑。",
		icon: "simple-icons:git",
		category: "tools",
		level: "advanced",
		experience: { years: 3, months: 0 },
		color: "#F05032",
	},
	{
		id: "latex",
		name: "LaTeX",
		description:
			"论文、讲义、复习手册、Beamer。Overleaf 协作与本地 latexmk 两套都用。",
		icon: "simple-icons:latex",
		category: "tools",
		level: "advanced",
		experience: { years: 3, months: 0 },
		color: "#008080",
	},
	{
		id: "obsidian",
		name: "Obsidian",
		description:
			"研究日志、看板、dataviewjs 仪表盘。整个工作流的中枢，专注统计的数据源也在这儿。",
		icon: "simple-icons:obsidian",
		category: "tools",
		level: "advanced",
		experience: { years: 1, months: 0 },
		color: "#7C3AED",
	},
	{
		id: "ai-assisted-research",
		name: "AI 辅助研究",
		description:
			"把 Claude Code 当研究助理：文献扫描、实验脚手架、自动化脚本、防造假的证据链检查。",
		icon: "simple-icons:anthropic",
		category: "tools",
		level: "advanced",
		experience: { years: 0, months: 9 },
		color: "#D97757",
	},

	// ── 研究之外 ──
	{
		id: "photography",
		name: "摄影",
		description: "日出和街头。富士，选片修片有一套自己的流程。",
		icon: "material-symbols:photo-camera",
		category: "other",
		level: "intermediate",
		experience: { years: 3, months: 0 },
		color: "#EC4899",
	},
	{
		id: "english",
		name: "英语",
		description:
			"全英环境学习四年，日常跟读、影子练习，看英文视频做双语笔记。",
		icon: "material-symbols:translate",
		category: "other",
		level: "advanced",
		experience: { years: 4, months: 0 },
		color: "#0891B2",
	},
	{
		id: "drawing-video",
		name: "画画与剪视频",
		description: "Procreate 画画，剪映剪视频。放松用的，不求产出。",
		icon: "material-symbols:brush",
		category: "other",
		level: "beginner",
		experience: { years: 2, months: 0 },
		color: "#A855F7",
	},
];
