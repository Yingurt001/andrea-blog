// 日记数据配置
// 用于管理日记页面的数据

export interface DiaryItem {
	id: number;
	content: string;
	date: string;
	images?: string[];
	location?: string;
	mood?: string;
	tags?: string[];
}

// 日记数据
const diaryData: DiaryItem[] = [
	{
		id: 1,
		content:
			"博客终于搭好了！折腾了一下午，从选主题到配置部署，虽然过程中遇到了一些小问题，但最终看到自己的博客跑起来的那一刻还是很开心的。希望自己能坚持写下去。",
		date: "2026-03-30T20:00:00Z",
		mood: "🎉",
		tags: ["博客", "里程碑"],
	},
];

// 获取日记列表（按时间倒序）
export const getDiaryList = (limit?: number) => {
	const sortedData = [...diaryData].sort(
		(a, b) => new Date(b.date).getTime() - new Date(a.date).getTime(),
	);

	if (limit && limit > 0) {
		return sortedData.slice(0, limit);
	}

	return sortedData;
};

// 获取所有标签
export const getAllTags = () => {
	const tags = new Set<string>();
	diaryData.forEach((item) => {
		if (item.tags) {
			item.tags.forEach((tag) => tags.add(tag));
		}
	});
	return Array.from(tags).sort();
};
