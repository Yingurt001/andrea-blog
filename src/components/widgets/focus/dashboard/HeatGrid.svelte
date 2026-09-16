<script lang="ts">
	// 热力图网格的 Svelte 版，总览（26 周）和年视图（整年）共用。
	// 跟侧栏的 FocusHeatmap.astro 吃同一套 utils，同一天在三处深浅一致；
	// 多出来的只有「点一格跳到那天」。tooltip 靠 data-tip，由 FocusTip.astro 的全局脚本接管。
	import {
		type HeatCell,
		cellStyle,
		formatTooltip,
		levelOf,
	} from "../../../../utils/focus-heatmap";

	interface Props {
		cells: HeatCell[];
		maxMinutes: number;
		weeks: number;
		gap?: string;
		monthLabels?: { col: number; label: string }[];
		onPick?: (date: string) => void;
	}
	const {
		cells,
		maxMinutes,
		weeks,
		gap = "3px",
		monthLabels,
		onPick,
	}: Props = $props();
</script>

<div class="focus-heatmap">
	{#if monthLabels}
		<div
			class="grid mb-1 relative h-4"
			style="gap:{gap};grid-template-columns:repeat({weeks},1fr)"
		>
			{#each monthLabels as m (m.col)}
				<span
					class="text-[10px] text-black/40 dark:text-white/40 absolute"
					style="left:calc({m.col} / {weeks} * 100%)">{m.label}</span
				>
			{/each}
		</div>
	{/if}
	<div
		class="grid grid-flow-col"
		style="gap:{gap};grid-template-rows:repeat(7,1fr);grid-template-columns:repeat({weeks},1fr)"
	>
		{#each cells as c (c.date)}
			{#if c.pad}
				<div class="aspect-square"></div>
			{:else}
				<button
					type="button"
					class="aspect-square w-full rounded-[2px] focus-cell cursor-pointer"
					style={cellStyle(levelOf(c.minutes, maxMinutes))}
					data-tip={formatTooltip(c)}
					aria-label={formatTooltip(c)}
					onclick={() => onPick?.(c.date)}
				></button>
			{/if}
		{/each}
	</div>
</div>
