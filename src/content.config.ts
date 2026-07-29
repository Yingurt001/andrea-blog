import { glob } from "astro/loaders";
import { z } from "astro/zod";
import { defineCollection } from "astro:content";

/* ------------------------------------------------------------------
 * 宽容型 frontmatter 解析
 *
 * 手写 frontmatter 的常见失误——标题写成纯数字（YAML 读成 number）、
 * tags 写成一行字符串、draft 写成 yes、published 忘了写——不应该让整个
 * Vercel 构建挂掉。下面这些 preprocess 在校验前先把输入掰正，掰不动的
 * 才落到默认值。
 *
 * 代价：掰正是静默的。`title: 7.30` 会被 YAML 读成 7.3，这里只能拿到
 * 7.3，还原不出末尾的 0。所以配套有 `pnpm check:content`
 * （scripts/lint-frontmatter.mjs）在推送前把这类可疑写法**警告**出来，
 * 它只提醒、不拦截。构建照过，问题看得见。
 * ------------------------------------------------------------------ */

const pad = (n: number) => String(n).padStart(2, "0");

const toDateString = (d: Date) =>
	`${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;

/** 任意标量 → string。YAML 会把 7.29 读成 number、2026-07-29 读成 Date、no 读成 false。 */
const looseString = (fallback = "") =>
	z.preprocess((v) => {
		if (v === undefined || v === null) return fallback;
		if (v instanceof Date) return toDateString(v);
		if (typeof v === "string") return v;
		if (typeof v === "number" || typeof v === "boolean") return String(v);
		return fallback;
	}, z.string());

/** 任意输入 → Date。写错或漏写时退回构建当天，而不是让构建失败。 */
const looseDate = (fallback: () => Date | undefined) =>
	z.preprocess((v) => {
		if (v instanceof Date && !Number.isNaN(v.getTime())) return v;
		if (typeof v === "string" || typeof v === "number") {
			const parsed = new Date(v);
			if (!Number.isNaN(parsed.getTime())) return parsed;
		}
		return fallback();
	}, z.date());

/** 任意输入 → boolean。接受 yes/no/on/off/1/0 这类 YAML 常见写法。 */
const looseBoolean = (fallback: boolean) =>
	z.preprocess((v) => {
		if (typeof v === "boolean") return v;
		if (typeof v === "number") return v !== 0;
		if (typeof v === "string") {
			const s = v.trim().toLowerCase();
			if (["true", "yes", "on", "1"].includes(s)) return true;
			if (["false", "no", "off", "0", ""].includes(s)) return false;
		}
		return fallback;
	}, z.boolean());

/** 任意输入 → string[]。单个标量当成一项；逗号分隔的字符串拆开。 */
const looseStringArray = () =>
	z.preprocess((v) => {
		if (v === undefined || v === null) return [];
		const scalarToString = (item: unknown): string => {
			if (item instanceof Date) return toDateString(item);
			if (typeof item === "string") return item.trim();
			if (typeof item === "number" || typeof item === "boolean")
				return String(item);
			return "";
		};
		if (Array.isArray(v)) return v.map(scalarToString).filter(Boolean);
		if (typeof v === "string")
			return v
				.split(/[,，]/)
				.map((s) => s.trim())
				.filter(Boolean);
		const single = scalarToString(v);
		return single ? [single] : [];
	}, z.array(z.string()));

/** 任意输入 → number | undefined。字符串数字也接受。 */
const looseNumber = () =>
	z.preprocess((v) => {
		if (typeof v === "number" && !Number.isNaN(v)) return v;
		if (typeof v === "string" && v.trim() !== "") {
			const parsed = Number(v);
			if (!Number.isNaN(parsed)) return parsed;
		}
		return undefined;
	}, z.number().optional());

const postsCollection = defineCollection({
	loader: glob({ pattern: "**/*.md", base: "./src/content/posts" }),
	schema: z.object({
		title: looseString("无标题"),
		published: looseDate(() => new Date()),
		updated: looseDate(() => undefined).optional(),
		draft: looseBoolean(false),
		description: looseString(""),
		image: looseString(""),
		tags: looseStringArray(),
		category: looseString(""),
		lang: looseString(""),
		pinned: looseBoolean(false),
		comment: looseBoolean(true),
		priority: looseNumber(),
		author: looseString(""),
		sourceLink: looseString(""),
		licenseName: looseString(""),
		licenseUrl: looseString(""),

		/* Page encryption fields */
		encrypted: looseBoolean(false),
		password: looseString(""),
		passwordHint: looseString(""),

		/* Posts alias */
		alias: looseString("").optional(),

		/* Custom permalink - 自定义固定链接，优先级高于 alias */
		permalink: looseString("").optional(),

		/* For internal use */
		prevTitle: looseString(""),
		prevSlug: looseString(""),
		nextTitle: looseString(""),
		nextSlug: looseString(""),
	}),
});
const specCollection = defineCollection({
	loader: glob({ pattern: "**/*.md", base: "./src/content/spec" }),
	schema: z.object({}),
});
export const collections = {
	posts: postsCollection,
	spec: specCollection,
};
