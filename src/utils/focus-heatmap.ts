// 专注热力图的格子计算。整页 /focus 和侧栏 widget 共用这一份 ——
// 两边各写一套的话，同一天会因为窗口长度不同落进不同的深浅档，
// 看起来像两处数据对不上，而其实只是归一化基准不一样。

export interface HeatCell {
	date: string;
	minutes: number;
}

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
		cells.push({ date: keyOf(d), minutes: days[keyOf(d)] ?? 0 });
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
