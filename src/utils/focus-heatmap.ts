// 专注热力图的格子计算。整页 /focus 和侧栏 widget 共用这一份 ——
// 两边各写一套的话，同一天会因为窗口长度不同落进不同的深浅档，
// 看起来像两处数据对不上，而其实只是归一化基准不一样。

export interface HeatCell {
	date: string;
	minutes: number;
	/** 当天各项目的分钟数，按多到少排好序。数据源已脱敏，这里拿到的就能直接显示。 */
	projects: [string, number][];
}

/** "YYYY-MM-DD" -> 项目名 -> 分钟数 */
export type DayProjects = Record<string, Record<string, number>>;

export interface Heatmap {
	cells: HeatCell[];
	maxMinutes: number;
	monthLabels: { col: number; label: string }[];
}

// level 0（当天没记录）走中性底色，其余四档靠 opacity 分级。
// 不用 rgba()：主题的 --primary 是 oklch()，rgba() 吃不下它。
export const LEVEL_OPACITY = [0, 0.3, 0.55, 0.75, 1];

const pad2 = (n: number) => String(n).padStart(2, "0");
const keyOf = (d: Date) =>
	`${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}`;

/**
 * 拼出「最近 weeks 周」的格子：7 行（周日~周六）× weeks 列。
 * maxMinutes 取全量 days 的最大值而不是窗口内的最大值 —— 窗口只决定
 * 看得见多长一段，不该决定颜色深浅，否则同一天在 14 周视图和 26 周视图里
 * 会是两种颜色。
 */
export function buildHeatmap(
	days: Record<string, number>,
	weeks: number,
	dayProjects: DayProjects = {},
): Heatmap {
	const today = new Date();
	today.setHours(0, 0, 0, 0);
	const endOfWeek = new Date(today);
	endOfWeek.setDate(endOfWeek.getDate() + (6 - endOfWeek.getDay())); // 补到本周六
	const start = new Date(endOfWeek);
	start.setDate(start.getDate() - weeks * 7 + 1);

	const cells: HeatCell[] = [];
	for (let i = 0; i < weeks * 7; i++) {
		const d = new Date(start);
		d.setDate(d.getDate() + i);
		const k = keyOf(d);
		// dayProjects 默认空对象：同步脚本更新前的旧快照没有这个字段，
		// 那种情况下 tooltip 退化成「日期 + 总时长」，不至于整页报错。
		const byProject = dayProjects[k] ?? {};
		cells.push({
			date: k,
			minutes: days[k] ?? 0,
			projects: Object.entries(byProject).sort((a, b) => b[1] - a[1]),
		});
	}

	const monthLabels: { col: number; label: string }[] = [];
	let lastMonth = -1;
	for (let w = 0; w < weeks; w++) {
		const m = Number(cells[w * 7].date.slice(5, 7));
		if (m !== lastMonth) {
			monthLabels.push({ col: w, label: `${m}月` });
			lastMonth = m;
		}
	}

	const all = Object.values(days);
	return {
		cells,
		maxMinutes: Math.max(1, ...all),
		monthLabels,
	};
}

export function levelOf(minutes: number, maxMinutes: number): number {
	if (!minutes) return 0;
	const ratio = minutes / maxMinutes;
	if (ratio > 0.75) return 4;
	if (ratio > 0.5) return 3;
	if (ratio > 0.25) return 2;
	return 1;
}

export function cellStyle(level: number): string {
	return level === 0
		? "background:var(--btn-regular-bg)"
		: `background:var(--primary);opacity:${LEVEL_OPACITY[level]}`;
}

const WEEKDAY = ["周日", "周一", "周二", "周三", "周四", "周五", "周六"];

/** 398 -> "6 小时 38 分"；不足一小时只给分钟，免得满屏「0 小时 x 分」。 */
export function fmtDuration(minutes: number): string {
	const h = Math.floor(minutes / 60);
	const m = minutes % 60;
	if (!h) return `${m} 分钟`;
	return m ? `${h} 小时 ${m} 分` : `${h} 小时`;
}

/**
 * 组装 tooltip 的文字。整页和侧栏共用，在构建时算好塞进格子的 data-tip，
 * 浏览器端只负责定位显示。返回值里的 \n 由 .focus-tip 的 white-space:pre-line 断行。
 */
export function formatTooltip(cell: HeatCell): string {
	const [y, m, d] = cell.date.split("-").map(Number);
	const head = `${m}月${d}日 ${WEEKDAY[new Date(y, m - 1, d).getDay()]}`;
	if (!cell.minutes) return `${head} · 没有记录`;

	// cell.projects 已按分钟从多到少排好，形如 [["GyroBN", 302], ["创业", 93]]。
	// 旧快照没有项目分解时这里是空数组，自然退回到只显示总时长。
	//
	// 全列，不截断也不滤掉几分钟的零头：少列一项，这几行之和就不等于首行的总时长，
	// 而那个对不上会让人怀疑整份数据，不是怀疑 tooltip 少显示了。一天最多 4 个项目，
	// 全列也就 5 行。
	const lines = cell.projects.map(
		([name, minutes]) => `${name} ${fmtDuration(minutes)}`,
	);

	return [`${head} · ${fmtDuration(cell.minutes)}`, ...lines].join("\n");
}
