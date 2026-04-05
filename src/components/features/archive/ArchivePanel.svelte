<script lang="ts">
	import I18nKey from "@i18n/i18nKey";
	import { i18n } from "@i18n/translation";
	import { onMount } from "svelte";

	export let tags: string[];
	export let categories: string[];
	export let sortedPosts: Post[] = [];

	const params = new URLSearchParams(window.location.search);
	tags = params.has("tag") ? params.getAll("tag") : [];
	categories = params.has("category") ? params.getAll("category") : [];
	const uncategorized = params.get("uncategorized");

	interface Post {
		id: string;
		url?: string; // 预计算的文章 URL
		data: {
			title: string;
			tags: string[];
			category?: string;
			published: Date;
			alias?: string;
			permalink?: string; // 自定义固定链接
		};
	}

	interface Group {
		year: number;
		posts: Post[];
	}

	let groups: Group[] = [];

	// 分类色标配置：hue 值对应 oklch 色彩空间
	const categoryColors: Record<string, { hue: number; label: string }> = {
		"技术": { hue: 145, label: "技术" },   // 绿色
		"生活": { hue: 45, label: "生活" },     // 橙色
		"研究": { hue: 260, label: "研究" },    // 紫色
		"科研": { hue: 220, label: "科研" },    // 蓝色
	};

	function getCategoryStyle(category?: string) {
		if (!category || !categoryColors[category]) {
			return { bg: "oklch(0.9 0.03 0)", text: "oklch(0.45 0.03 0)", label: "未分类" };
		}
		const { hue, label } = categoryColors[category];
		return {
			bg: `oklch(0.92 0.04 ${hue})`,
			text: `oklch(0.45 0.1 ${hue})`,
			label,
		};
	}

	function formatDate(date: Date) {
		const month = (date.getMonth() + 1).toString().padStart(2, "0");
		const day = date.getDate().toString().padStart(2, "0");
		return `${month}-${day}`;
	}

	function formatTag(tagList: string[]) {
		return tagList.map((t) => `#${t}`).join(" ");
	}

	onMount(async () => {
		let filteredPosts: Post[] = sortedPosts;

		if (tags.length > 0) {
			filteredPosts = filteredPosts.filter(
				(post) =>
					Array.isArray(post.data.tags) &&
					post.data.tags.some((tag) => tags.includes(tag)),
			);
		}

		if (categories.length > 0) {
			filteredPosts = filteredPosts.filter(
				(post) =>
					post.data.category &&
					categories.includes(post.data.category),
			);
		}

		if (uncategorized) {
			filteredPosts = filteredPosts.filter((post) => !post.data.category);
		}

		// 按发布时间倒序排序，确保不受置顶影响
		filteredPosts = filteredPosts
			.slice()
			.sort(
				(a, b) =>
					b.data.published.getTime() - a.data.published.getTime(),
			);

		const grouped = filteredPosts.reduce(
			(acc, post) => {
				const year = post.data.published.getFullYear();
				if (!acc[year]) {
					acc[year] = [];
				}
				acc[year].push(post);
				return acc;
			},
			{} as Record<number, Post[]>,
		);

		const groupedPostsArray = Object.keys(grouped).map((yearStr) => ({
			year: Number.parseInt(yearStr, 10),
			posts: grouped[Number.parseInt(yearStr, 10)],
		}));

		groupedPostsArray.sort((a, b) => b.year - a.year);

		groups = groupedPostsArray;
	});
</script>

<div class="card-base px-8 py-6">
	{#each groups as group, gi}
		<div>
			<!-- 年份标题 -->
			<div
				class="archive-fade-in flex flex-row w-full items-center h-[3.75rem]"
				style="--delay: {gi * 60}ms;"
			>
				<div
					class="w-[15%] md:w-[10%] text-2xl font-bold text-right text-75"
				>
					{group.year}
				</div>
				<div class="w-[15%] md:w-[10%]">
					<div
						class="h-3 w-3 bg-none rounded-full outline outline-[var(--primary)] mx-auto
                  -outline-offset-[2px] z-50 outline-3"
					></div>
				</div>
				<div class="w-[70%] md:w-[80%] text-left text-50">
					{group.posts.length}
					{i18n(
						group.posts.length === 1
							? I18nKey.postCount
							: I18nKey.postsCount,
					)}
				</div>
			</div>

			{#each group.posts as post, pi}
				<a
					href={post.url || `/posts/${post.id}/`}
					aria-label={post.data.title}
					class="archive-fade-in group btn-plain !block h-10 w-full rounded-lg hover:text-[initial]"
					style="--delay: {gi * 60 + (pi + 1) * 25}ms;"
				>
					<div
						class="flex flex-row justify-start items-center h-full"
					>
						<!-- date -->
						<div
							class="w-[15%] md:w-[10%] text-sm text-right text-50
                     transition-colors duration-100 ease-out group-hover:text-[var(--primary)]"
						>
							{formatDate(post.data.published)}
						</div>

						<!-- dot and line -->
						<div
							class="w-[15%] md:w-[10%] relative dash-line h-full flex items-center"
						>
							<div
								class="transition-all duration-100 ease-out mx-auto w-1 h-1 rounded group-hover:h-5
                       bg-[oklch(0.5_0.05_var(--hue))] group-hover:bg-[var(--primary)]
                       outline outline-4 z-50
                       outline-[var(--card-bg)]
                       group-hover:outline-[var(--btn-plain-bg-hover)]
                       group-active:outline-[var(--btn-plain-bg-active)]"
							></div>
						</div>

						<!-- category badge + post title -->
						<div
							class="w-[70%] md:max-w-[65%] md:w-[65%] text-left font-bold
                     group-hover:translate-x-1 transition-transform duration-100 ease-out group-hover:text-[var(--primary)]
                     text-75 pr-8 whitespace-nowrap overflow-ellipsis overflow-hidden flex items-center gap-1.5"
						>
							<span
								class="inline-flex items-center shrink-0 text-xs font-medium px-1.5 py-0 rounded
                       transition-[transform,box-shadow] duration-100 ease-out group-hover:shadow-sm group-hover:scale-105"
								style="background-color: {getCategoryStyle(post.data.category).bg}; color: {getCategoryStyle(post.data.category).text};"
							>
								{getCategoryStyle(post.data.category).label}
							</span>
							<span class="overflow-hidden overflow-ellipsis">{post.data.title}</span>
						</div>

						<!-- tag list -->
						<div
							class="hidden md:block md:w-[15%] text-left text-sm
                     whitespace-nowrap overflow-ellipsis overflow-hidden text-30
                     transition-colors duration-100 ease-out group-hover:text-50"
						>
							{formatTag(post.data.tags)}
						</div>
					</div>
				</a>
			{/each}
		</div>
	{/each}
</div>

<style>
	@keyframes archive-enter {
		from {
			opacity: 0;
			transform: translateY(6px);
		}
		to {
			opacity: 1;
			transform: translateY(0);
		}
	}

	.archive-fade-in {
		animation: archive-enter 0.35s cubic-bezier(0.16, 1, 0.3, 1) both;
		animation-delay: var(--delay, 0ms);
		will-change: transform, opacity;
	}

	/* 减少动画偏好的用户：直接显示 */
	@media (prefers-reduced-motion: reduce) {
		.archive-fade-in {
			animation: none;
			opacity: 1;
		}
	}
</style>
