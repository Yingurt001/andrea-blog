<script lang="ts">
	// /focus 页的主体：一排标签切 总览 / 日 / 周 / 月 / 年。
	// 四个带时间的视图共用一个光标（某一天的 key）：从周视图点进某天，日视图就停在那天；
	// 再切到月视图，看的还是那天所在的月。光标默认停在最后一个有记录的天，而不是「今天」——
	// 这个值只由数据决定，服务端渲染和浏览器水合两边必然一致，不会因为构建日期和访问日期
	// 不同而对不上。
	import { onMount } from "svelte";
	import { focusStats } from "../../../../data/focus-stats";
	import {
		type FocusView,
		firstActiveDay,
		lastActiveDay,
	} from "../../../../utils/focus-agg";
	import DayView from "./DayView.svelte";
	import MonthView from "./MonthView.svelte";
	import Overview from "./Overview.svelte";
	import WeekView from "./WeekView.svelte";
	import YearView from "./YearView.svelte";

	interface Props {
		/** 构建期定下的「今天」，只用来锚定总览热力图的右端和高亮今天 */
		todayKey: string;
	}
	const { todayKey }: Props = $props();

	const TABS: { id: FocusView; label: string }[] = [
		{ id: "overview", label: "总览" },
		{ id: "day", label: "日" },
		{ id: "week", label: "周" },
		{ id: "month", label: "月" },
		{ id: "year", label: "年" },
	];
	const isView = (s: string): s is FocusView => TABS.some((t) => t.id === s);

	const lastKey = lastActiveDay(focusStats.days) ?? todayKey;
	// 翻页的边界：最早有记录的那天 ~ max(最后有记录的那天, 今天)。今天没记录也允许翻到，
	// 不然「今天还没开始专注」这种最常见的情况会被挡在最后一个记录日。
	const minKey = firstActiveDay(focusStats.days) ?? todayKey;
	const maxKey = lastKey > todayKey ? lastKey : todayKey;

	let view: FocusView = $state("overview");
	let cursor = $state(lastKey);

	function go(v: FocusView, key?: string) {
		view = v;
		if (key) cursor = key;
		// 点格子切走后那个格子没了，浮层会留在原地
		document.dispatchEvent(new Event("focus-tip:hide"));
		// 标签写进 URL hash，方便直接分享「周视图」这种链接；不入历史栈，免得后退键在标签间打转
		history.replaceState(
			null,
			"",
			v === "overview" ? location.pathname + location.search : `#${v}`,
		);
	}

	onMount(() => {
		const h = location.hash.slice(1);
		if (isView(h)) view = h;
	});

	const viewProps = $derived({
		cursor,
		todayKey,
		minKey,
		maxKey,
		lastKey,
		onGo: go,
	});
</script>

<div
	class="flex gap-1 p-1 rounded-xl bg-[var(--btn-regular-bg)] w-fit mb-6"
	role="tablist"
	aria-label="专注视图"
>
	{#each TABS as t (t.id)}
		<button
			type="button"
			role="tab"
			aria-selected={view === t.id}
			class="px-3.5 py-1.5 rounded-lg text-sm font-medium transition {view ===
			t.id
				? 'bg-[var(--primary)] text-white shadow-sm'
				: 'text-black/60 dark:text-white/60 hover:text-[var(--primary)]'}"
			onclick={() => go(t.id)}
		>
			{t.label}
		</button>
	{/each}
</div>

{#if view === "overview"}
	<Overview {todayKey} onPick={(k) => go("day", k)} />
{:else if view === "day"}
	<DayView {...viewProps} />
{:else if view === "week"}
	<WeekView {...viewProps} />
{:else if view === "month"}
	<MonthView {...viewProps} />
{:else}
	<YearView {...viewProps} />
{/if}
