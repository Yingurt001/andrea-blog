/**
 * 开发环境专用：在 Finder 里打开 public/images。
 *
 * 为什么要走服务端：浏览器禁止 https/localhost 页面跳 file:// 链接，
 * 页面自己打不开本地文件夹。但 `astro dev` 的服务端就跑在本机上，
 * 所以让它代为执行 `open`。
 *
 * ⚠️ 站点是 output: "static"，构建时 Astro 会**执行**所有端点来预渲染。
 * 下面 import.meta.env.DEV 这道闸不能去掉，否则 Vercel 构建机会真去跑 open。
 * 生产环境该端点只会产出一个静态的 {"ok":false} JSON。
 *
 * 目标路径写死不接受参数：一来 output: "static" 下查询参数传不进端点
 * （2026-07-29 实测，四个不同的 ?dir= 返回完全相同），二来没有外部输入
 * 就没有目录穿越和命令注入的空间。
 */
import { execFile } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

function json(body: unknown, status = 200) {
	return new Response(JSON.stringify(body), {
		status,
		headers: { "Content-Type": "application/json" },
	});
}

export async function GET() {
	if (!import.meta.env.DEV) {
		return json({ ok: false, reason: "该接口仅在开发环境可用" }, 404);
	}

	const target = path.resolve("public/images");

	if (!fs.existsSync(target)) {
		return json({ ok: false, reason: `找不到 ${target}` }, 404);
	}

	const opener =
		process.platform === "darwin"
			? "open"
			: process.platform === "win32"
				? "explorer"
				: "xdg-open";

	return await new Promise<Response>((resolve) => {
		// 参数数组、不经 shell
		execFile(opener, [target], (err) => {
			if (err) {
				resolve(json({ ok: false, reason: err.message }, 500));
			} else {
				resolve(json({ ok: true, opened: target }));
			}
		});
	});
}
