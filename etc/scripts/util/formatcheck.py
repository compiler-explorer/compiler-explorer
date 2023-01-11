import sys
import urllib.request
import urllib.parse
import json
import subprocess
import os.path


from typing import Set

OWNER_REPO = ""
ACCESS_TOKEN = ""
USER_AGENT = ""


def get(entity: str, query: dict = None) -> dict:
    if query is None:
        query = {}
    path = entity
    if query:
        querystring = urllib.parse.urlencode(query)
        path += f"?{querystring}"
    print(f"Getting {path}")
    req = urllib.request.Request(
        f"https://api.github.com/{path}",
        None,
        {
            "User-Agent": USER_AGENT,
            "Authorization": f"token {ACCESS_TOKEN}",
            "Accept": "application/vnd.github.v3+json",
        },
    )
    result = urllib.request.urlopen(req)
    # It's ok not to check for error codes here. We'll throw either way
    return json.loads(result.read())


def paginated_get(entity: str, query: dict = None) -> [dict]:
    if query is None:
        query = {}
    result = []
    results_per_page = 50
    query["page"] = 1
    query["per_page"] = results_per_page
    while True:
        current_page_results = get(entity, query)
        result.extend(current_page_results)
        if len(current_page_results) == results_per_page:
            query["page"] += 1
        else:
            break
    return result


def list_open_prs(stale_label: str = None) -> [dict]:
    prs = paginated_get(f"repos/{OWNER_REPO}/pulls", {"state": "open"})
    if stale_label is not None:
        return [pr for pr in prs if not any(label["name"] == stale_label for label in pr["labels"])]
    return prs


def list_pr_files(pr: dict) -> [dict]:
    return paginated_get(f'repos/{OWNER_REPO}/pulls/{pr["number"]}/files')


def list_modified_paths_in_pr(pr: dict) -> Set[str]:
    pr_paths = list_pr_files(pr)
    filtered = {x["filename"] for x in pr_paths if x["status"] == "modified"}
    return filtered


def list_files_under_vc() -> Set[str]:
    output = subprocess.check_output(
        ["git", "ls-tree", "-r", "main", "--name-only"]
    ).decode("utf-8")
    paths = set(output.splitlines())
    return paths


def make_file_formateable(path: str):
    try:
        with open(path, "r+") as f:
            content = ["/**\n", " * @prettier\n", " */\n"]
            file_contents = f.readlines()
            if file_contents[0:3] != content:
                content.extend(file_contents)
                f.seek(0)
                f.writelines(content)
                f.truncate()
    except:
        print(f"Error making {path} formatable")


def main():
    # In case you want to save the result to avoid extra API calls
    use_file = False
    if not use_file or not os.path.isfile("./paths.json"):
        current_prs = list_open_prs(stale_label="likely-stale")
        modified_paths = set()
        for PR in current_prs:
            modified_paths.update(list_modified_paths_in_pr(PR))
        current_git_files = list_files_under_vc()
        untouched_paths = list(current_git_files - modified_paths)
        if use_file:
            with open("./paths.json", "w") as f:
                json.dump(untouched_paths, f)
    else:
        with open("./paths.json", "r") as f:
            untouched_paths = json.load(f)
    js_files = {x for x in untouched_paths if x.endswith(".js")}
    for path in js_files:
        make_file_formateable(path)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: {os.path.basename(__file__)} OWNER/REPO ACCESS_TOKEN USER_AGENT")
    else:
        OWNER_REPO = sys.argv[1]
        ACCESS_TOKEN = sys.argv[2]
        USER_AGENT = sys.argv[3]
        main()
