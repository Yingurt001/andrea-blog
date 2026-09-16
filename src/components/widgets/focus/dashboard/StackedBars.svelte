<script lang="ts">
	// 按项目堆叠的柱状图：周视图 7 根（一天一根）、年视图 12 根（一月一根）共用。
	// 每根柱子按「这段时间里的最高那根」归一化，不用全量最大值——
	// 周视图里最高的一天要顶满，不然清淡的一周七根柱子全趴在底下看不出节奏。
	import { colorFor, fmtShort } from "../../../../utils/focus-agg";

	export interface Column {
		id: string;
		label: string;
		sub?: string;
		minutes: number;
		parts: [string, number][];
		tip: string;
		today?: boolean;
	}
	interface Props {
		columns: Column[];
		colors: Record<string, string>;
		height?: number;
		onPick?: (id: string) => void;
	}
	const { columns, colors, height = 160, onPick }: Props = $props();
	const max = $derived(Math.max(1, ...columns.map((c) => c.minutes)));
</script>

<div
	class="grid gap-1.5 sm:gap-2.5"
	style="grid-template-columns:repeat({columns.length},minmax(0,1fr))"
>
	{#each columns as c (c.id)}
		<button
			type="button"
			class="group flex flex-col items-center min-w-0 rounded-lg focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--primary)]"
			onclick={() => onPick?.(c.id)}
			data-tip={c.tip}
			aria-label={c.tip}
		>
			<div
				class="text-[10px] mb-1 h-3 tabular-nums text-black/40 dark:text-white/40"
			>
				{c.minutes ? fmtShort(c.minutes) : ""}
			</div>
			<div
				class="w-full flex flex-col justify-end border-b border-[var(--line-divider)]"
				style="height:{height}px"
			>
				{#if c.minutes}
					<div
						class="w-full flex flex-col-reverse rounded-t-md overflow-hidden transition group-hover:brightness-110"
						style="height:{(c.minutes / max) * 100}%"
					>
						{#each c.parts as [name, m] (name)}
							<div
								style="height:{(m / c.minutes) *
									100}%;background:{colorFor(name, colors)}"
							></div>
						{/each}
					</div>
				{/if}
			</div>
			<div
				class="mt-1.5 text-xs {c.today
					? 'text-[var(--primary)] font-semibold'
					: 'text-black/60 dark:text-white/60'}"
			>
				{c.label}
			</div>
			{#if c.sub}<div
					class="text-[10px] text-black/35 dark:text-white/35"
				>
					{c.sub}
				</div>{/if}
		</button>
	{/each}
</div>
