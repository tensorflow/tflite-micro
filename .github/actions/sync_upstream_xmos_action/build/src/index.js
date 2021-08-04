"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const github_1 = require("@actions/github");
const core_1 = require("@actions/core");
const io_1 = require("@actions/io");
const dependencies = ['git'];
const token = core_1.getInput('token');
const git_baseURL = github_1.context.serverUrl || 'https://github.com';
const upstream = 'xmos/tflite-micro';
const octokit = github_1.getOctokit(token);
function binExists(name) {
    return io_1.which(name)
        .then(() => Promise.resolve(true))
        .catch(() => Promise.resolve(false));
}
async function hasDeps(deps) {
    return !(await Promise.all(deps.map(dep => binExists(dep)))).includes(false);
}
async function init() {
    if (!(await hasDeps(dependencies)))
        throw new Error('Dependencies not resolved. One of:' +
            `[${dependencies.join(', ')}] was unavailable.`);
}
async function run() {
    console.log(process.env);
}
init()
    .then(run)
    .catch(e => {
    core_1.setFailed(e.message());
});
//# sourceMappingURL=index.js.map