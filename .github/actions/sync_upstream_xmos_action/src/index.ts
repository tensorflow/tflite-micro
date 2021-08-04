import {context, getOctokit} from '@actions/github';
import {setFailed, getInput} from '@actions/core';
import {which} from '@actions/io';

const dependencies = ['git'];
const token = getInput('token');
const git_baseURL = context.serverUrl || 'https://github.com';
const upstream = 'xmos/tflite-micro';
const octokit = getOctokit(token);

function binExists(name: string): Promise<boolean> {
  return which(name)
    .then(() => Promise.resolve(true))
    .catch(() => Promise.resolve(false));
}

async function hasDeps(deps: Array<string>): Promise<boolean> {
  return !(await Promise.all(deps.map(dep => binExists(dep)))).includes(false);
}

async function init() {
  if (!(await hasDeps(dependencies)))
    throw new Error(
      'Dependencies not resolved. One of:' +
        `[${dependencies.join(', ')}] was unavailable.`
    );
}

async function run() {
  console.log(process.env);
}

init()
  .then(run)
  .catch(e => {
    setFailed(e.message());
  });
