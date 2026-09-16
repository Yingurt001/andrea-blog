// 日 / 周 / 月 / 年 四个视图共用的聚合函数。
// 输入永远是 focus-stats.ts 里那三份东西（days / dayProjects / segments），
// 这里只做「按一段日期把它们加起来」，不碰任何时区换算——切片和时区在导出端已经做完。

import { type DayProjects, keyOf, pad2, parseKey } from "./focus-heatmap";

export type FocusView = "overview" | "day" | "week" | "month" | "year";

export interface Segment {
	d: string;
	s: number;
	e: number;
	m: number;
	p: string;
}

/** 周一到周日的中文名，周视图和月视图的表头都从周一起（跟侧栏日历一致），热力图那边是周日起 */
export const WEEKDAY_MON = ["一", "二", "三", "四", "五", "六", "日"];
const WEEKDAY_SUN = ["周日", "周一", "周二", "周三", "周四", "周五", "周六"];

export function addDays(key: string, n: number): string {
	const d = parseKey(key);
	d.setDate(d.getDate() + n);
	return keyOf(d);
}

/** "2026-09" 往前后挪 n 个月，返回那个月的 1 号 */
export function addMonths(key: string, n: number): string {
	const d = parseKey(key);
	d.setDate(1);
	d.setMonth(d.getMonth() + n);
	return keyOf(d);
}

export function addYears(key: string, n: number): string {
	const y = Number(key.slice(0, 4)) + n;
	return `${y}-01-01`;
}

/** "2026-09-16" -> "9月16日 周三" */
export function cnDate(key: string): string {
	const d = parseKey(key);
	return `${d.getMonth() + 1}月${d.getDate()}日 ${WEEKDAY_SUN[d.getDay()]}`;
}

/** "9月14日 – 9月20日" */
export function cnRange(a: string, b: string): string {
	const da = parseKey(a);
	const db = parseKey(b);
	return `${da.getMonth() + 1}月${da.getDate()}日 – ${db.getMonth() + 1}月${db.getDate()}日`;
}

/** ISO 周号（周一起，含 1 月 4 日的那周是第 1 周） */
export function isoWeek(key: string): number {
	const d = parseKey(key);
	const t = new Date(Date.UTC(d.getFullYear(), d.getMonth(), d.getDate()));
	const day = t.getUTCDay() || 7;
	t.setUTCDate(t.getUTCDate() + 4 - day);
	const yStart = Date.UTC(t.getUTCFullYear(), 0, 1);
	return Math.ceil(((t.getTime() - yStart) / 86400000 + 1) / 7);
}

/** 分钟 -> 时间轴刻度 "09:05"；1440 写成 "24:00"，那是跨日切片的右端 */
export function hm(min: number): string {
	const h = Math.floor(min / 60);
	return `${pad2(h)}:${pad2(Math.round(min % 60))}`;
}

/** 紧凑写法，给放不下「x 小时 y 分」的地方用：45 -> "45m"，398 -> "6.6h" */
export function fmtShort(min: number): string {
	if (min < 60) return `${min}m`;
	const h = min / 60;
	return `${h >= 10 ? h.toFixed(0) : h.toFixed(1)}h`;
}

/** 包含 anchor 的那一周，周一到周日七个 key */
export function weekKeys(anchor: string): string[] {
	const d = parseKey(anchor);
	const offset = (d.getDay() + 6) % 7; // 周一 0 … 周日 6
	const mon = addDays(anchor, -offset);
	return Array.from({ length: 7 }, (_, i) => addDays(mon, i));
}

/** 某个月的全部 key，按顺序 */
export function monthKeys(anchor: string): string[] {
	const first = parseKey(anchor);
	first.setDate(1);
	const n = new Date(first.getFullYear(), first.getMonth() + 1, 0).getDate();
	return Array.from({ length: n }, (_, i) => addDays(keyOf(first), i));
}

/** 月历网格：每行 7 格、周一起，月外的位置是 null */
export function monthGrid(anchor: string): (string | null)[][] {
	const keys = monthKeys(anchor);
	const lead = (parseKey(keys[0]).getDay() + 6) % 7;
	const flat: (string | null)[] = [...Array(lead).fill(null), ...keys];
	while (flat.length % 7) flat.push(null);
	const rows: (string | null)[][] = [];
	for (let i = 0; i < flat.length; i += 7) rows.push(flat.slice(i, i + 7));
	return rows;
}

export function yearMonthKeys(anchor: string): string[] {
	const y = anchor.slice(0, 4);
	return Array.from({ length: 12 }, (_, i) => `${y}-${pad2(i + 1)}-01`);
}

export function sumDays(days: Record<string, number>, keys: string[]): number {
	return keys.reduce((s, k) => s + (days[k] ?? 0), 0);
}

export function activeCount(
	days: Record<string, number>,
	keys: string[],
): number {
	return keys.filter((k) => (days[k] ?? 0) > 0).length;
}

/** 这段日期里各项目的合计，多到少排。 */
export function projectTotals(
	dayProjects: DayProjects,
	keys: string[],
): [string, number][] {
	const acc: Record<string, number> = {};
	for (const k of keys) {
		for (const [name, m] of Object.entries(dayProjects[k] ?? {}))
			acc[name] = (acc[name] ?? 0) + m;
	}
	return Object.entries(acc).sort((a, b) => b[1] - a[1]);
}

/** 时间段按天分桶，日视图一天只取一桶，不用每次都扫全量 */
export function groupSegments(segments: Segment[]): Record<string, Segment[]> {
	const by: Record<string, Segment[]> = {};
	for (const s of segments) (by[s.d] ??= []).push(s);
	return by;
}

export function firstActiveDay(days: Record<string, number>): string | null {
	const ks = Object.keys(days)
		.filter((k) => days[k] > 0)
		.sort();
	return ks[0] ?? null;
}

export function lastActiveDay(days: Record<string, number>): string | null {
	const ks = Object.keys(days)
		.filter((k) => days[k] > 0)
		.sort();
	return ks[ks.length - 1] ?? null;
}

// 博客自己的一套色：亮、饱和、彼此拉得开。私有面板那套色板前 12 个是亮色，之后的项目
// 只能拿深棕深绿之类的备用档，搬到博客上跟浅色主题打架，所以这里不再沿用导出的 projectColors。
// 分配规则：按历史总时长排名，第一名拿第一个颜色。整份快照算一次，日/周/月/年四个视图
// 里同一个项目永远同色。「其他」是被黑名单合并进来的杂项，固定给浅灰，不占亮色。
const DOPAMINE = [
	"#ff6b9d", // 樱桃粉
	"#ffb443", // 杏橙
	"#7c6cff", // 紫罗兰
	"#5dd39e", // 薄荷绿
	"#4d96ff", // 天蓝
	"#ffd93d", // 柠檬黄
	"#ff7e5f", // 珊瑚
	"#3ed2c3", // 松石
	"#c77dff", // 丁香紫
	"#00bbf9", // 湖蓝
	"#f15bb5", // 洋红
	"#ff9f1c", // 橘
];
const OTHER_NAME = "其他";
const OTHER_COLOR = "#c4c9d4";

export function buildPalette(dayProjects: DayProjects): Record<string, string> {
	const total: Record<string, number> = {};
	for (const byProject of Object.values(dayProjects)) {
		for (const [name, m] of Object.entries(byProject))
			total[name] = (total[name] ?? 0) + m;
	}
	const ranked = Object.keys(total)
		.filter((n) => n !== OTHER_NAME)
		.sort((a, b) => total[b] - total[a] || a.localeCompare(b));
	const palette: Record<string, string> = { [OTHER_NAME]: OTHER_COLOR };
	ranked.forEach((n, i) => {
		palette[n] = DOPAMINE[i % DOPAMINE.length];
	});
	return palette;
}

/** 没进色板的名字（理论上不会有）按名字哈希兜底，同名永远同色。 */
export function colorFor(name: string, colors: Record<string, string>): string {
	if (colors[name]) return colors[name];
	let h = 0;
	for (const ch of name) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
	return DOPAMINE[h % DOPAMINE.length];
}

/**
 * 连续活跃天数。longest 是历史上最长的一段；current 是截到 upTo 那天还没断的一段
 * （upTo 当天没记录的话 current 就是 0）。年视图的「最长连续」和总览的「当前连续」都吃这个。
 */
export function streaks(
	days: Record<string, number>,
	upTo: string,
): { longest: number; current: number } {
	// 「活跃」的门槛跟「活跃天数」那个数字块一致：当天有记录（> 0 分钟）就算。
	// 两处口径不一样的话，「活跃 12 天」和「最长连续」会对不上，读者会先怀疑数据。
	// 想改成「至少 25 分钟才算」之类，改这一行的判断，两个数字块会一起变。
	const active = Object.keys(days)
		.filter((k) => days[k] > 0 && k <= upTo)
		.sort();
	let longest = 0;
	let run = 0;
	let prev: string | null = null;
	for (const k of active) {
		run = prev && addDays(prev, 1) === k ? run + 1 : 1;
		if (run > longest) longest = run;
		prev = k;
	}
	// current 只在「upTo 当天就有记录」时才算还没断；最后一段是昨天结束的也算断了。
	const current = prev === upTo ? run : 0;
	return { longest, current };
}
