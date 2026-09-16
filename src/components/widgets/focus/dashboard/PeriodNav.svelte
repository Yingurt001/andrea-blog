<script lang="ts">
	// 「‹ 标题 ›」翻页条。箭头到数据边界就灰掉：没记录的年份翻过去一片空白，不如不让翻。
	interface Props {
		title: string;
		sub?: string;
		canPrev: boolean;
		canNext: boolean;
		onShift: (n: number) => void;
		/** 给了就在右边多一个快捷按钮（「最近」「今天」之类） */
		jumpLabel?: string;
		onJump?: () => void;
	}
	const { title, sub, canPrev, canNext, onShift, jumpLabel, onJump }: Props =
		$props();
	const btn =
		"w-8 h-8 rounded-lg flex items-center justify-center bg-[var(--btn-regular-bg)] hover:bg-[var(--btn-regular-bg-hover)] text-black/70 dark:text-white/70 disabled:opacity-30 disabled:cursor-not-allowed transition";
</script>

<div class="flex items-center justify-center gap-2 mb-5">
	<button
		type="button"
		class={btn}
		disabled={!canPrev}
		onclick={() => onShift(-1)}
		aria-label="上一个"
	>
		<svg
			width="18"
			height="18"
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			stroke-width="2.2"
			stroke-linecap="round"
			stroke-linejoin="round"><path d="M15 6l-6 6 6 6" /></svg
		>
	</button>
	<div class="text-center min-w-[10rem]">
		<div class="font-semibold text-black/85 dark:text-white/85">
			{title}
		</div>
		{#if sub}<div class="text-xs text-black/40 dark:text-white/40 mt-0.5">
				{sub}
			</div>{/if}
	</div>
	<button
		type="button"
		class={btn}
		disabled={!canNext}
		onclick={() => onShift(1)}
		aria-label="下一个"
	>
		<svg
			width="18"
			height="18"
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			stroke-width="2.2"
			stroke-linecap="round"
			stroke-linejoin="round"><path d="M9 6l6 6-6 6" /></svg
		>
	</button>
	{#if onJump && jumpLabel}
		<button
			type="button"
			class="ml-1 px-2.5 h-8 rounded-lg text-xs font-medium bg-[var(--btn-regular-bg)] text-[var(--primary)] hover:bg-[var(--btn-regular-bg-hover)] transition"
			onclick={onJump}
		>
			{jumpLabel}
		</button>
	{/if}
</div>
