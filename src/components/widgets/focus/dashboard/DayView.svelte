<script lang="ts">
	// 日视图：回答「这天什么时候在干什么」。24 小时时间轴 + 项目份额 + 逐段明细。
	import { focusStats } from "../../../../data/focus-stats";
	import {
		type FocusView,
		addDays,
		cnDate,
		colorFor,
		groupSegments,
		hm,
	} from "../../../../utils/focus-agg";
	import { cellOf, fmtDuration } from "../../../../utils/focus-heatmap";
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
	const colors = focusStats.projectColors ?? {};
	// 旧快照没有 segments 字段（同步按钮更新前导出的），或者同步脚本关了 EXPORT_TIMELINE：
	// 那就没有时间轴，只画项目份额，别的照常。
	const hasTimeline = (focusStats.segments ?? []).length > 0;
	const segsByDay = groupSegments(focusStats.segments ?? []);

	const cell = $derived(cellOf(cursor, days, dayProjects));
	const segs = $derived(segsByDay[cursor] ?? []);
	const longest = $derived(segs.reduce((m, s) => Math.max(m, s.m), 0));
	const tiles = $derived(
		hasTimeline
			? [
					{
						value: (cell.minutes / 60).toFixed(1),
						label: "当天时长（小时）",
					},
					{ value: segs.length, label: "专注次数" },
					{
						value: (longest / 60).toFixed(1),
						label: "最长一段（小时）",
					},
				]
			: [
					{
						value: (cell.minutes / 60).toFixed(1),
						label: "当天时长（小时）",
					},
					{ value: cell.projects.length, label: "涉及项目" },
				],
	);
	const year = $derived(cursor.slice(0, 4));
	const TICKS = ["0:00", "6:00", "12:00", "18:00", "24:00"];
</script>

<PeriodNav
	title={`${year} 年 ${cnDate(cursor)}`}
	sub={cursor === todayKey ? "今天" : undefined}
	canPrev={cursor > minKey}
	canNext={cursor < maxKey}
	onShift={(n) => onGo("day", addDays(cursor, n))}
	jumpLabel={cursor === lastKey ? undefined : "最近一天"}
	onJump={() => onGo("day", lastKey)}
/>

<StatTiles items={tiles} />

{#if !cell.minutes}
	<p class="text-center text-black/40 dark:text-white/40 text-sm py-10">
		这天没有专注记录
	</p>
{:else}
	{#if hasTimeline}
		<div class="text-xs text-black/40 dark:text-white/40 mb-2">
			这天的时间轴
		</div>
		<div
			class="relative h-9 rounded-lg bg-[var(--btn-regular-bg)] overflow-hidden"
		>
			{#each [6, 12, 18] as h (h)}
				<div
					class="absolute top-0 bottom-0 w-px bg-black/5 dark:bg-white/10"
					style="left:{(h / 24) * 100}%"
				></div>
			{/each}
			{#each segs as s, i (i)}
				<div
					class="absolute top-0 bottom-0 rounded-[3px] hover:brightness-110 transition"
					style="left:{(s.s / 1440) * 100}%;width:{Math.max(
						0.35,
						((s.e - s.s) / 1440) * 100,
					)}%;background:{colorFor(s.p, colors)}"
					data-tip={`${hm(s.s)} ~ ${hm(s.e)} · ${s.p} · ${fmtDuration(s.m)}`}
				></div>
			{/each}
		</div>
		<div
			class="flex justify-between mt-1 mb-6 text-[10px] tabular-nums text-black/35 dark:text-white/35"
		>
			{#each TICKS as t (t)}<span>{t}</span>{/each}
		</div>
	{/if}

	<ProjectBars rows={cell.projects} {colors} title="这天的时间去了哪" />

	{#if hasTimeline && segs.length}
		<div class="text-xs text-black/40 dark:text-white/40 mt-6 mb-2">
			明细
		</div>
		<ul class="flex flex-col gap-1.5">
			{#each segs as s, i (i)}
				<li
					class="flex items-center gap-3 px-3 py-2 rounded-lg bg-[var(--btn-regular-bg)] text-sm"
				>
					<span
						class="w-2 h-2 rounded-full shrink-0"
						style="background:{colorFor(s.p, colors)}"
					></span>
					<span
						class="w-28 shrink-0 tabular-nums text-xs text-black/50 dark:text-white/50"
						>{hm(s.s)} ~ {hm(s.e)}</span
					>
					<span
						class="flex-1 truncate text-black/80 dark:text-white/80"
						>{s.p}</span
					>
					<span
						class="tabular-nums text-xs text-black/50 dark:text-white/50"
						>{fmtDuration(s.m)}</span
					>
				</li>
			{/each}
		</ul>
	{/if}
{/if}
