<script lang="ts">
	// 年视图：12 根月柱看趋势，整年热力图看密度，底下是全年份额。
	import { focusStats } from "../../../../data/focus-stats";
	import {
		type FocusView,
		buildPalette,
		activeCount,
		addYears,
		monthKeys,
		projectTotals,
		streaks,
		sumDays,
		yearMonthKeys,
	} from "../../../../utils/focus-agg";
	import {
		buildYearHeatmap,
		fmtDuration,
	} from "../../../../utils/focus-heatmap";
	import HeatGrid from "./HeatGrid.svelte";
	import PeriodNav from "./PeriodNav.svelte";
	import ProjectBars from "./ProjectBars.svelte";
	import StackedBars, { type Column } from "./StackedBars.svelte";
	import StatTiles from "./StatTiles.svelte";

	interface Props {
		cursor: string;
		todayKey: string;
		minKey: string;
		maxKey: string;
		lastKey: string;
		onGo: (view: FocusView, key?: string) => void;
	}
	const { cursor, todayKey, minKey, maxKey, lastKey, onGo }: Props = $props();

	const { days, dayProjects } = focusStats;
	const colors = buildPalette(focusStats.dayProjects);

	const year = $derived(cursor.slice(0, 4));
	const months = $derived(yearMonthKeys(cursor));
	const keys = $derived(months.flatMap(monthKeys));
	const total = $derived(sumDays(days, keys));
	const active = $derived(activeCount(days, keys));
	// 连续天数只在这一年里数，跨年那段归到哪年都别扭，干脆两边各算各的
	const yearDays = $derived(
		Object.fromEntries(
			Object.entries(days).filter(([k]) => k.startsWith(year)),
		),
	);
	const st = $derived(streaks(yearDays, `${year}-12-31`));
	const heat = $derived(buildYearHeatmap(days, dayProjects, Number(year)));

	const columns: Column[] = $derived(
		months.map((mk, i) => {
			const ks = monthKeys(mk);
			const minutes = sumDays(days, ks);
			const parts = projectTotals(dayProjects, ks);
			const head = `${year}年${i + 1}月 · ${minutes ? fmtDuration(minutes) : "没有记录"}`;
			return {
				id: mk,
				label: `${i + 1}月`,
				minutes,
				parts,
				tip: [
					head,
					...parts.map(([n, m]) => `${n} ${fmtDuration(m)}`),
				].join("\n"),
				today: mk.slice(0, 7) === todayKey.slice(0, 7),
			};
		}),
	);
	const tiles = $derived([
		{ value: (total / 60).toFixed(1), label: "全年合计（小时）" },
		{ value: active, label: "活跃天数" },
		{ value: st.longest, label: "最长连续（天）" },
		{
			value: active ? (total / active / 60).toFixed(1) : "0",
			label: "活跃日均（小时）",
		},
	]);
</script>

<PeriodNav
	title={`${year} 年`}
	canPrev={year > minKey.slice(0, 4)}
	canNext={year < maxKey.slice(0, 4)}
	onShift={(n) => onGo("year", addYears(cursor, n))}
	jumpLabel={year === lastKey.slice(0, 4) ? undefined : "最近一年"}
	onJump={() => onGo("year", lastKey)}
/>

<StatTiles items={tiles} />

{#if !total}
	<p class="text-center text-black/40 dark:text-white/40 text-sm py-10">
		这一年没有专注记录
	</p>
{:else}
	<StackedBars
		{columns}
		{colors}
		height={140}
		onPick={(k) => onGo("month", k)}
	/>
	<p
		class="text-[11px] text-black/35 dark:text-white/35 mt-2 mb-6 text-right"
	>
		点柱子看那个月
	</p>

	<div class="text-xs text-black/40 dark:text-white/40 mb-2">全年每一天</div>
	<div class="overflow-x-auto pb-2 mb-6">
		<div class="min-w-[720px]">
			<HeatGrid
				cells={heat.cells}
				maxMinutes={heat.maxMinutes}
				weeks={heat.cells.length / 7}
				monthLabels={heat.monthLabels}
				gap="2px"
				onPick={(k) => onGo("day", k)}
			/>
		</div>
	</div>

	<ProjectBars
		rows={projectTotals(dayProjects, keys)}
		{colors}
		title="这一年的时间去了哪"
	/>
{/if}
