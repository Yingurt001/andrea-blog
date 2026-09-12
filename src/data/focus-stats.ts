// 专注统计快照
// 由 Obsidian 私有笔记 `_dashboard/editor/focus.md` 的「📤 同步到博客公开统计页」按钮
// 覆盖写入，不要手动改这个文件的数值——改了下次同步会被覆盖。
// 只导出「每天总时长」，不带项目名（求职/IELTS 之类偏私人，默认不公开）。

export interface FocusStats {
	totalMinutes: number;
	totalSessions: number;
	activeDays: number;
	days: Record<string, number>; // "YYYY-MM-DD" -> 当天分钟数
	updatedAt: string | null;
}

export const focusStats: FocusStats = {
  "totalMinutes": 2358,
  "totalSessions": 34,
  "activeDays": 7,
  "days": {
    "2026-09-04": 156,
    "2026-09-05": 522,
    "2026-09-07": 262,
    "2026-09-08": 234,
    "2026-09-09": 655,
    "2026-09-10": 480,
    "2026-09-11": 49
  },
  "updatedAt": "2026-09-11T15:37:03.079Z"
};
