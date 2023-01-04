# This script when called from an error handler in a workflow will
# create an issue. 
# 
# The title will be the name of the workflow and the 
# body will contain a link to the run that failed. The issue will be
# tagged with a label that can be used to identify it as automation 
# generated. 
# 
# If the issue already exists, the script will add a comment
# to the issue with the time and date of the error and a link to the
# run that failed. 

from github import Github
from datetime import datetime
import os

TOKEN = os.environ['TOKEN']
REPO_NAME = os.environ['REPO']
WORKFLOW = os.environ['WORKFLOW']
FLAG_LABEL = os.environ['FLAG_LABEL']
RUN_NUMBER = os.environ['RUN_NUMBER']
RUN_ID = os.environ['RUN_ID']

def get_tagged_issues(repo, flag_label, workflow):
    issues = repo.get_issues(state='open', labels=[flag_label])
    tagged_issues =[]
    for issue in issues:
        if workflow in issue.body:
            tagged_issues.append(issue)
    return(tagged_issues)

def create_issue(repo, repo_name, flag_label, workflow, run_number, run_id):
    run_link = f"http://github.com/{repo_name}/actions/runs/{run_id}"
    body_string = f"{workflow} [run number {run_number}]({run_link}) failed.\n"
    body_string += "Please examine the run itself for details.\n\n"
    body_string += "This issue has been automatically generated for "
    body_string += "notification purposes."

    title_string = f"{workflow} Scheduled Run Failed"
    new_issue = repo.create_issue(title = title_string, body = body_string, 
                                  labels=[flag_label])
    return(new_issue)


def add_comment(issue, repo_name, run_id, run_number):
    run_link = f"http://github.com/{repo_name}/actions/runs/{run_id}"
    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    msg_string = f"Error reoccurred: {dt_string}\n"
    msg_string += f"[Run number: {run_number}]({run_link})\n"
    issue.create_comment(msg_string) 
    return()

if __name__ == "__main__":
    
    g = Github(TOKEN)
    repo = g.get_repo(REPO_NAME)
    tagged_issues = get_tagged_issues(repo, FLAG_LABEL, WORKFLOW)
    if not tagged_issues:
        create_issue(repo, REPO_NAME, FLAG_LABEL, WORKFLOW, RUN_NUMBER, RUN_ID)
    else:
        for issue in tagged_issues:
            add_comment(issue, REPO_NAME, RUN_ID, RUN_NUMBER)
    for issue in tagged_issues:
        print(issue.number)
        print(issue.closed_at)
    