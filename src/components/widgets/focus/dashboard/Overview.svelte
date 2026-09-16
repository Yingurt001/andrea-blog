<script lang="ts">
	// 总览 = 原来那一页：四个数字 + 最近 26 周热力图。点格子跳到那天的日视图。
	import { focusStats } from "../../../../data/focus-stats";
	import { streaks } from "../../../../utils/focus-agg";
	import {
		LEVEL_OPACITY,
		buildHeatmap,
		cellStyle,
	} from "../../../../utils/focus-heatmap";
	import HeatGrid from "./HeatGrid.svelte";
	import StatTiles from "./StatTiles.svelte";

	interface Props {
		todayKey: string;
		onPick: (date: string) => void;
	}
	const { todayKey, onPick }: Props = $props();

	const WEEKS = 26;
	const { totalMinutes, totalSessions, activeDays, days, dayProjects } =
		focusStats;
	const heat = buildHeatmap(days, WEEKS, dayProjects, todayKey);
	const st = streaks(days, todayKey);
	const tiles = [
		{ value: (totalMinutes / 60).toFixed(1), label: "总时长（小时）" },
		{ value: totalSessions, label: "专注次数" },
		{ value: activeDays, label: "活跃天数" },
		{ value: st.longest, label: "最长连续（天）" },
	];
</script>

<StatTiles items={tiles} />

<div class="overflow-x-auto pb-2">
	<div class="min-w-[560px]">
		<HeatGrid
			cells={heat.cells}
			maxMinutes={heat.maxMinutes}
			weeks={WEEKS}
			monthLabels={heat.monthLabels}
			{onPick}
		/>
	</div>
</div>

<div
	class="flex items-center gap-2 mt-4 text-xs text-black/40 dark:text-white/40"
>
	<span>少</span>
	{#each LEVEL_OPACITY as _, level (level)}
		<span
			class="w-3 h-3 rounded-[2px] inline-block"
			style={cellStyle(level)}
		></span>
	{/each}
	<span>多</span>
	<span class="ml-auto">点任意一格看那天</span>
</div>
