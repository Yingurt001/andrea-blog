<script lang="ts">
	// 月视图：月历，每格写时长、底下一条按项目分色的细条。深浅跟热力图同一套档位。
	import { focusStats } from "../../../../data/focus-stats";
	import {
		type FocusView,
		buildPalette,
		WEEKDAY_MON,
		activeCount,
		addMonths,
		colorFor,
		fmtShort,
		monthGrid,
		monthKeys,
		projectTotals,
		sumDays,
	} from "../../../../utils/focus-agg";
	import {
		cellOf,
		formatTooltip,
		levelOf,
	} from "../../../../utils/focus-heatmap";
	import PeriodNav from "./PeriodNav.svelte";
	import ProjectBars from "./ProjectBars.svelte";
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
	const maxMinutes = Math.max(1, ...Object.values(days));

	const ym = $derived(cursor.slice(0, 7));
	const keys = $derived(monthKeys(cursor));
	const grid = $derived(monthGrid(cursor));
	const total = $derived(sumDays(days, keys));
	const active = $derived(activeCount(days, keys));
	const tiles = $derived([
		{ value: (total / 60).toFixed(1), label: "本月合计（小时）" },
		{ value: active, label: "活跃天数" },
		{
			value: active ? (total / active / 60).toFixed(1) : "0",
			label: "活跃日均（小时）",
		},
	]);

	// 底色不用 opacity（会把格子里的字一起淡掉），改成把主色按档位混进底色里。
	// 深的两档字换成白色，不然主色底上黑字看不清。
	const MIX = [0, 28, 50, 72, 100];
	const bgOf = (level: number) =>
		level === 0
			? "var(--btn-regular-bg)"
			: `color-mix(in oklch, var(--primary) ${MIX[level]}%, var(--btn-regular-bg))`;
</script>

<PeriodNav
	title={`${ym.slice(0, 4)} 年 ${Number(ym.slice(5, 7))} 月`}
	canPrev={ym > minKey.slice(0, 7)}
	canNext={ym < maxKey.slice(0, 7)}
	onShift={(n) => onGo("month", addMonths(cursor, n))}
	jumpLabel={ym === lastKey.slice(0, 7) ? undefined : "最近一月"}
	onJump={() => onGo("month", lastKey)}
/>

<StatTiles items={tiles} />

<div
	class="grid grid-cols-7 gap-1 sm:gap-1.5 mb-1 text-center text-[11px] text-black/40 dark:text-white/40"
>
	{#each WEEKDAY_MON as w (w)}<span>{w}</span>{/each}
</div>
<div class="grid grid-cols-7 gap-1 sm:gap-1.5">
	{#each grid as row, r (r)}
		{#each row as k, c (k ?? `pad-${r}-${c}`)}
			{#if k}
				{@const cell = cellOf(k, days, dayProjects)}
				{@const level = levelOf(cell.minutes, maxMinutes)}
				<button
					type="button"
					class="relative rounded-lg p-1.5 min-h-[52px] sm:min-h-[64px] flex flex-col justify-between text-left transition hover:brightness-105 focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--primary)] {level >=
					3
						? 'text-white'
						: 'text-black/70 dark:text-white/70'}"
					style="background:{bgOf(level)}"
					data-tip={formatTooltip(cell)}
					aria-label={formatTooltip(cell)}
					onclick={() => onGo("day", k)}
				>
					<span
						class="text-[11px] sm:text-xs leading-none {k ===
						todayKey
							? 'font-bold underline decoration-2 underline-offset-2'
							: ''}">{Number(k.slice(8, 10))}</span
					>
					{#if cell.minutes}
						<span
							class="text-[10px] sm:text-xs font-medium tabular-nums self-end leading-none"
							>{fmtShort(cell.minutes)}</span
						>
						<span
							class="flex h-1 mt-1 rounded-full overflow-hidden gap-px"
						>
							{#each cell.projects as [name, m] (name)}
								<span
									style="width:{(m / cell.minutes) *
										100}%;background:{colorFor(
										name,
										colors,
									)}"
								></span>
							{/each}
						</span>
					{:else}
						<span class="h-1 mt-1"></span>
					{/if}
				</button>
			{:else}
				<div></div>
			{/if}
		{/each}
	{/each}
</div>
<p class="text-[11px] text-black/35 dark:text-white/35 mt-2 mb-6 text-right">
	点日期看那天
</p>

<ProjectBars
	rows={projectTotals(dayProjects, keys)}
	{colors}
	title="这个月的时间去了哪"
/>
