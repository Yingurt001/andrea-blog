#!/usr/bin/env node
/**
 * 文章 frontmatter 体检器 —— 只警告，不拦截（永远 exit 0）。
 *
 * src/content.config.ts 里的 schema 已经是宽容型的：写歪了会被静默掰正，
 * 构建不会失败。但"掰正"有代价，最典型的是 YAML 会把 `title: 7.30` 读成
 * 数字 7.3，末尾的 0 再也找不回来——页面上就显示成 7.3。
 *
 * 这个脚本读的是 frontmatter 的**原始文本**（而不是解析后的值），所以能看见
 * 解析器会破坏掉的东西，把它们提示出来。
 *
 * 真正会让构建挂掉的（比如 YAML 语法本身就错），交给 `astro sync` 兜底。
 */

import fs from "fs";
import path from "path";

const POSTS_DIR = path.join(process.cwd(), "src/content/posts");

/**
 * YAML 里会被读成非字符串的裸值。
 *
 * 注意 Astro 用的是 YAML 1.2 core schema：只有 true/false 是布尔，
 * yes/no/on/off 一律留作字符串（这点和 YAML 1.1 不同，2026-07-29 实测确认过，
 * 别想当然按 1.1 加回去）。
 */
const NUMERIC = /^-?\d+(\.\d+)?$/;
const YAML_BOOL = /^(true|false)$/i;
const YAML_NULL = /^(null|~)$/i;
const BARE_DATE = /^\d{4}-\d{1,2}-\d{1,2}$/;

/** 这些字段 schema 要求是字符串，裸值会被掰正 */
const STRING_FIELDS = [
	"title",
	"description",
	"category",
	"author",
	"lang",
	"image",
	"licenseName",
	"licenseUrl",
	"sourceLink",
	"passwordHint",
];
const DATE_FIELDS = ["published", "updated"];

function isQuoted(v) {
	return /^".*"$/.test(v) || /^'.*'$/.test(v);
}

/** 提取 frontmatter 原始文本块 */
function extractFrontmatter(raw) {
	if (!raw.startsWith("---")) return null;
	const end = raw.indexOf("\n---", 3);
	if (end === -1) return null;
	return raw.slice(raw.indexOf("\n") + 1, end);
}

function lintFile(filePath) {
	const raw = fs.readFileSync(filePath, "utf-8");
	const fm = extractFrontmatter(raw);
	const issues = [];

	if (fm === null) {
		issues.push({
			level: "err",
			msg: "没有找到 frontmatter（文件开头必须是一行 --- ，结尾再一行 ---）",
		});
		return issues;
	}

	// 只看顶层 key（行首无缩进）
	const seen = new Map();
	for (const line of fm.split("\n")) {
		const m = line.match(/^([A-Za-z_][\w-]*):\s*(.*?)\s*$/);
		if (m) seen.set(m[1], m[2]);
	}

	if (!seen.has("title")) {
		issues.push({
			level: "warn",
			msg: "缺 title —— 页面会显示成「无标题」",
		});
	}
	if (!seen.has("published")) {
		issues.push({
			level: "warn",
			msg: "缺 published —— 会退回成「构建当天」，意味着每次部署日期都会变",
		});
	}

	for (const field of STRING_FIELDS) {
		if (!seen.has(field)) continue;
		const v = seen.get(field);
		if (v === "" || isQuoted(v)) continue;

		if (NUMERIC.test(v)) {
			const parsed = String(Number(v));
			const lost = parsed !== v;
			issues.push({
				level: lost ? "err" : "warn",
				msg: lost
					? `${field}: ${v} 会被 YAML 读成数字，显示成「${parsed}」——写成 ${field}: "${v}"`
					: `${field}: ${v} 会被 YAML 读成数字（显示无变化，但建议写成 ${field}: "${v}"）`,
			});
		} else if (YAML_BOOL.test(v)) {
			issues.push({
				level: "err",
				msg: `${field}: ${v} 会被 YAML 读成布尔值，显示成「${v.toLowerCase()}」——写成 ${field}: "${v}"`,
			});
		} else if (YAML_NULL.test(v)) {
			issues.push({
				level: "err",
				msg: `${field}: ${v} 会被 YAML 读成空值——写成 ${field}: "${v}"`,
			});
		} else if (BARE_DATE.test(v)) {
			issues.push({
				level: "warn",
				msg: `${field}: ${v} 会被 YAML 读成日期对象——写成 ${field}: "${v}" 更保险`,
			});
		}
	}

	for (const field of DATE_FIELDS) {
		if (!seen.has(field)) continue;
		const v = seen.get(field).replace(/^["']|["']$/g, "");
		if (v === "") continue;
		if (Number.isNaN(new Date(v).getTime())) {
			issues.push({
				level: "err",
				msg: `${field}: ${v} 解析不成日期 —— 会退回成构建当天。推荐写 ${field}: 2026-07-29`,
			});
		}
	}

	return issues;
}

function main() {
	if (!fs.existsSync(POSTS_DIR)) {
		console.log("跳过 frontmatter 体检：找不到 src/content/posts");
		return;
	}

	const files = fs
		.readdirSync(POSTS_DIR, { recursive: true })
		.filter((f) => typeof f === "string" && f.endsWith(".md"))
		.sort();

	let errCount = 0;
	let warnCount = 0;
	const report = [];

	for (const rel of files) {
		const issues = lintFile(path.join(POSTS_DIR, rel));
		if (issues.length === 0) continue;
		report.push({ rel, issues });
		for (const i of issues) {
			if (i.level === "err") errCount++;
			else warnCount++;
		}
	}

	if (report.length === 0) {
		console.log(`✔ frontmatter 体检通过（${files.length} 篇文章，无可疑写法）`);
		return;
	}

	console.log("");
	console.log("frontmatter 体检报告（只提醒，不会阻止构建或推送）");
	console.log("────────────────────────────────────────────────────");
	for (const { rel, issues } of report) {
		console.log(`\n  ${rel}`);
		for (const i of issues) {
			console.log(`    ${i.level === "err" ? "✘" : "⚠"} ${i.msg}`);
		}
	}
	console.log("");
	console.log("────────────────────────────────────────────────────");
	console.log(
		`共 ${files.length} 篇，${errCount} 处会导致显示错误，${warnCount} 处建议改。`,
	);
	console.log("构建照常能过，改不改随你。");
}

main();
