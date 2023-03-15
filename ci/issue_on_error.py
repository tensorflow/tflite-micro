# issue_on_error_post.py
# Requires python 3.6+ and GitHub CLI in the environment.
#
# Creates or updates an issue on action failure related to a workflow. 
#
# Looks though REPO for open issues with FLAG_LABEL. If none,
# creates a new issue. If there is an open issue with FLAG_LABEL,
# looks for WORKFLOW in body. If none, creates an issue. If an
# issue for WORKFLOW exists, makes adds a comment.
#
# Requires the environment provide the variables in the block below.
# TOKEN must have access to update issues.
#
# If called with an optional PR_NUMBER and PR_LINK the issue will 
# include a link to the PR. 


from datetime import datetime
import os
import json
import subprocess

REPO_NAME = os.environ['REPO']
WORKFLOW = os.environ['WORKFLOW']
FLAG_LABEL = os.environ['FLAG_LABEL']
RUN_NUMBER = os.environ['RUN_NUMBER']
RUN_ID = os.environ['RUN_ID']
# optional variables
PR_NUMBER = os.getenv('PR_NUMBER')
PR_LINK = os.getenv('PR_LINK')

def get_tagged_issues(flag_label, workflow):
    issues = subprocess.check_output(["gh", "issue", "list",
                                        "--state", "open",
                                        "--label", flag_label,
                                        "--json", "title,number,body"], 
                                        encoding="utf-8")
    issues = json.loads(issues)
    tagged_issues =[]
    for issue in issues:
        if workflow in issue["body"]:
            tagged_issues.append(issue)
    return(tagged_issues)

def create_issue(flag_label, workflow, run_number, run_id, repo_name, pr_number, pr_link):

    run_link = f"http://github.com/{repo_name}/actions/runs/{run_id}"
    body_string = ""
    title_string = f"{workflow} CI Run Failed"
    if pr_number:
        body_string = f"PR {pr_number} ({pr_link}) had a CI failure: \n"
        title_string = f"PR #{pr_number} CI Run Failed"
    body_string += f"{workflow} [run number {run_number}]({run_link}) failed. \n\n"
    body_string += "This issue has been automatically generated for "
    body_string += "notification purposes."
    
    new_issue = subprocess.check_output(["gh", "issue", "create", 
                                        "--title", title_string,
                                        "--body", body_string,
                                        "--label", flag_label], 
                                        encoding="utf-8")
    return(new_issue)


def add_comment(issue_number, run_number, run_id, repo_name):
    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    run_link = f"http://github.com/{repo_name}/actions/runs/{run_id}"
    msg_string = f"Error reoccurred: {dt_string}\n"
    msg_string += f"[Run number: {run_number}]({run_link})\n"
    subprocess.run(["gh", "issue", "comment", issue_number, 
                    "--body", msg_string])
    return()

if __name__ == "__main__":
    tagged_issues = get_tagged_issues(FLAG_LABEL, WORKFLOW)

    # The logic catches the case where an issue exists for the workflow
    # but we are testing against a PR and want a created issue to link to the PR.
    # Otherwise, we just add a comment to the existing issue.
    if not tagged_issues:
        create_issue(FLAG_LABEL, WORKFLOW, RUN_NUMBER, RUN_ID, REPO_NAME, PR_NUMBER, PR_LINK)
    else:
        for issue in tagged_issues:
            if PR_NUMBER:
                if PR_LINK in issue["body"]:
                    add_comment(str(issue["number"]), RUN_NUMBER, RUN_ID, REPO_NAME)
                else:
                    create_issue(FLAG_LABEL, WORKFLOW, RUN_NUMBER, RUN_ID, REPO_NAME, 
                                PR_NUMBER, PR_LINK)
            else:
                add_comment(str(issue["number"]), RUN_NUMBER, RUN_ID, REPO_NAME)
    for issue in tagged_issues:
        print(issue["number"])
        
    



