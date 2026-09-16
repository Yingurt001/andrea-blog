<script lang="ts">
	// 周视图：一天一根柱子，看这周的节奏；跟上周比一眼看出是加了还是松了。
	import { focusStats } from "../../../../data/focus-stats";
	import {
		type FocusView,
		buildPalette,
		WEEKDAY_MON,
		activeCount,
		addDays,
		cnRange,
		isoWeek,
		projectTotals,
		sumDays,
		weekKeys,
	} from "../../../../utils/focus-agg";
	import { cellOf, formatTooltip } from "../../../../utils/focus-heatmap";
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

	const keys = $derived(weekKeys(cursor));
	const total = $derived(sumDays(days, keys));
	const active = $derived(activeCount(days, keys));
	const prevTotal = $derived(sumDays(days, weekKeys(addDays(cursor, -7))));
	// 上周是 0 的话百分比没意义（除零），显示成「—」
	const delta = $derived(
		prevTotal ? Math.round(((total - prevTotal) / prevTotal) * 100) : null,
	);
	const inLast = $derived(keys.includes(lastKey));

	const columns: Column[] = $derived(
		keys.map((k, i) => {
			const c = cellOf(k, days, dayProjects);
			return {
				id: k,
				label: WEEKDAY_MON[i],
				sub: String(Number(k.slice(8, 10))),
				minutes: c.minutes,
				parts: c.projects,
				tip: formatTooltip(c),
				today: k === todayKey,
			};
		}),
	);
	const tiles = $derived([
		{ value: (total / 60).toFixed(1), label: "本周合计（小时）" },
		{
			value: active ? (total / active / 60).toFixed(1) : "0",
			label: "活跃日均（小时）",
		},
		{
			value: delta === null ? "—" : `${delta > 0 ? "+" : ""}${delta}%`,
			label: "较上周",
		},
	]);
</script>

<PeriodNav
	title={cnRange(keys[0], keys[6])}
	sub={`${keys[0].slice(0, 4)} 年 第 ${isoWeek(keys[0])} 周`}
	canPrev={keys[0] > minKey}
	canNext={keys[6] < maxKey}
	onShift={(n) => onGo("week", addDays(cursor, n * 7))}
	jumpLabel={inLast ? undefined : "最近一周"}
	onJump={() => onGo("week", lastKey)}
/>

<StatTiles items={tiles} />

{#if !total}
	<p class="text-center text-black/40 dark:text-white/40 text-sm py-10">
		这周没有专注记录
	</p>
{:else}
	<StackedBars {columns} {colors} onPick={(k) => onGo("day", k)} />
	<p
		class="text-[11px] text-black/35 dark:text-white/35 mt-2 mb-6 text-right"
	>
		点柱子看那天
	</p>
	<ProjectBars
		rows={projectTotals(dayProjects, keys)}
		{colors}
		title="这周的时间去了哪"
	/>
{/if}
