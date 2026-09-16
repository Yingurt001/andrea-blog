<script lang="ts">
	// 「时间去了哪」：一行一个项目，条长 = 占这段时间的份额。
	import { colorFor } from "../../../../utils/focus-agg";
	import { fmtDuration } from "../../../../utils/focus-heatmap";

	interface Props {
		rows: [string, number][];
		colors: Record<string, string>;
		title?: string;
	}
	const { rows, colors, title }: Props = $props();
	const total = $derived(
		Math.max(
			1,
			rows.reduce((s, r) => s + r[1], 0),
		),
	);
</script>

{#if rows.length}
	<div>
		{#if title}<div class="text-xs text-black/40 dark:text-white/40 mb-2">
				{title}
			</div>{/if}
		<ul class="flex flex-col gap-2">
			{#each rows as [name, m] (name)}
				<li class="flex items-center gap-2.5 text-sm">
					<span
						class="w-2.5 h-2.5 rounded-sm shrink-0"
						style="background:{colorFor(name, colors)}"
					></span>
					<span
						class="w-24 sm:w-32 truncate text-black/75 dark:text-white/75"
						>{name}</span
					>
					<span
						class="flex-1 h-2 rounded-full bg-[var(--btn-regular-bg)] overflow-hidden"
					>
						<span
							class="block h-full rounded-full"
							style="width:{(m / total) *
								100}%;background:{colorFor(name, colors)}"
						></span>
					</span>
					<span
						class="w-24 text-right tabular-nums text-xs text-black/60 dark:text-white/60"
						>{fmtDuration(m)}</span
					>
					<span
						class="w-9 text-right tabular-nums text-xs text-black/35 dark:text-white/35"
						>{Math.round((m / total) * 100)}%</span
					>
				</li>
			{/each}
		</ul>
	</div>
{/if}
