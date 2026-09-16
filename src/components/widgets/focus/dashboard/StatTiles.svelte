<script lang="ts">
	// 顶部那排数字块。auto-fit 让 3 块和 4 块都能占满一行，手机上自动折成两列。
	interface Props {
		items: { value: string | number; label: string }[];
	}
	const { items }: Props = $props();
</script>

<div
	class="grid gap-3 sm:gap-4 mb-6"
	style="grid-template-columns:repeat(auto-fit,minmax(120px,1fr))"
>
	{#each items as it (it.label)}
		<div
			class="rounded-xl bg-[var(--btn-regular-bg)] p-3.5 sm:p-5 text-center"
		>
			<div
				class="text-xl sm:text-3xl font-bold text-[var(--primary)] tabular-nums"
			>
				{it.value}
			</div>
			<div
				class="text-xs sm:text-sm text-black/50 dark:text-white/50 mt-1"
			>
				{it.label}
			</div>
		</div>
	{/each}
</div>
