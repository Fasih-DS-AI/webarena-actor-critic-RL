"""Task definitions for the MiniWebArena environment.

Each task is an ordered sequence of *subgoals*. A subgoal is identified by the
``intent`` of the UI element that must be activated to satisfy it (e.g. the
subgoal ``search_btn`` is satisfied by clicking the element whose intent is
``search_btn``). This mirrors WebArena's "functional evaluators": a task is
complete only when the correct sequence of grounded element interactions has
been performed.

Encoding tasks as sequences of *intents* (rather than opaque task IDs) means a
policy can learn the transferable rule "activate the element matching the
current required intent", which generalises to held-out task *compositions* —
the evaluation tasks below recombine intents seen during training into novel
sequences, testing compositional generalisation rather than memorisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


# ---------------------------------------------------------------------------
# Intent vocabulary
# ---------------------------------------------------------------------------
# "Goal" intents are the correct element to activate for some subgoal.
# "Distractor" intents are plausible but wrong elements present on a page.
# "Trap" intents send the agent to a dead-end/error page (requires recovery).
# "back" is the recovery element shown on a dead-end page.

GOAL_INTENTS: List[str] = [
    # e-commerce (shop)
    "search_box", "search_btn", "filter", "item_link", "add_cart",
    "checkout", "confirm_order",
    # forum
    "forum_link", "new_thread", "thread_title", "thread_body", "submit_post",
    "thread_link", "reply_box",
    # gitlab
    "new_project", "project_name", "create_repo", "readme_file", "commit_btn",
    "project_link", "issues_tab", "new_issue", "issue_title", "submit_issue",
    # cms
    "new_page", "page_title", "page_content", "publish_btn", "page_link",
    "edit_btn",
]

DISTRACTOR_INTENTS: List[str] = ["home_link", "profile_link", "help_link", "footer_link"]
TRAP_INTENTS: List[str] = ["ad_popup", "logout_btn"]
RECOVERY_INTENT: str = "back"

# Master vocabulary (order is stable -> defines one-hot indices).
INTENTS: List[str] = GOAL_INTENTS + DISTRACTOR_INTENTS + TRAP_INTENTS + [RECOVERY_INTENT]
INTENT2ID = {name: i for i, name in enumerate(INTENTS)}
N_INTENTS = len(INTENTS)

SITES: List[str] = ["shop", "forum", "gitlab", "cms"]
SITE2ID = {name: i for i, name in enumerate(SITES)}


@dataclass(frozen=True)
class Task:
    """An ordered web-navigation task.

    Attributes:
        name: human-readable identifier.
        site: which of the four sites the task lives on.
        subgoals: ordered list of intents that must be activated in sequence.
        description: natural-language goal (used in the report / logging).
    """

    name: str
    site: str
    subgoals: List[str]
    description: str

    @property
    def length(self) -> int:
        return len(self.subgoals)


# ---------------------------------------------------------------------------
# Training tasks  (cover the entire goal-intent vocabulary at least once)
# ---------------------------------------------------------------------------
TRAIN_TASKS: List[Task] = [
    Task(
        name="shop_buy",
        site="shop",
        subgoals=["search_box", "search_btn", "filter", "item_link",
                  "add_cart", "checkout", "confirm_order"],
        description="Search for an item, filter results, add the cheapest to "
                    "the cart and complete checkout.",
    ),
    Task(
        name="forum_post",
        site="forum",
        subgoals=["forum_link", "new_thread", "thread_title", "thread_body",
                  "submit_post"],
        description="Open a forum and post a new discussion thread.",
    ),
    Task(
        name="forum_reply",
        site="forum",
        subgoals=["forum_link", "thread_link", "reply_box", "submit_post"],
        description="Open a forum, open a thread and post a reply.",
    ),
    Task(
        name="gitlab_repo",
        site="gitlab",
        subgoals=["new_project", "project_name", "create_repo", "readme_file",
                  "commit_btn"],
        description="Create a new GitLab project and commit a README file.",
    ),
    Task(
        name="gitlab_issue",
        site="gitlab",
        subgoals=["project_link", "issues_tab", "new_issue", "issue_title",
                  "submit_issue"],
        description="Open a project, go to issues and file a new issue.",
    ),
    Task(
        name="cms_page",
        site="cms",
        subgoals=["new_page", "page_title", "page_content", "publish_btn",
                  "page_link", "edit_btn"],
        description="Create and publish a CMS page, then open it for editing.",
    ),
]


# ---------------------------------------------------------------------------
# Held-out evaluation tasks  (novel recombinations of *seen* intents)
# ---------------------------------------------------------------------------
EVAL_TASKS: List[Task] = [
    Task(
        name="shop_quickbuy",
        site="shop",
        subgoals=["search_box", "search_btn", "item_link", "checkout",
                  "confirm_order"],
        description="Search for an item and buy it directly without filtering.",
    ),
    Task(
        name="forum_quickthread",
        site="forum",
        subgoals=["forum_link", "new_thread", "thread_title", "submit_post"],
        description="Open a forum and post a short thread (title only).",
    ),
    Task(
        name="gitlab_issue_quick",
        site="gitlab",
        subgoals=["project_link", "new_issue", "issue_title", "submit_issue"],
        description="Open a project and quickly file an issue.",
    ),
    Task(
        name="cms_edit",
        site="cms",
        subgoals=["page_link", "edit_btn", "page_content", "publish_btn"],
        description="Open an existing CMS page, edit it and republish.",
    ),
]

ALL_TASKS: List[Task] = TRAIN_TASKS + EVAL_TASKS


def _validate_intent_coverage() -> None:
    """Ensure every intent used in EVAL_TASKS is also present in TRAIN_TASKS.

    Held-out tasks must only recombine *seen* intents, otherwise generalisation
    is impossible (the network never receives gradients for unseen one-hot
    dimensions). This guards against accidentally introducing an unseen intent.
    """
    train_intents = {ig for t in TRAIN_TASKS for ig in t.subgoals}
    eval_intents = {ig for t in EVAL_TASKS for ig in t.subgoals}
    missing = eval_intents - train_intents
    assert not missing, f"EVAL tasks use intents never seen in TRAIN: {missing}"
    unknown = {ig for t in ALL_TASKS for ig in t.subgoals} - set(INTENTS)
    assert not unknown, f"Tasks reference intents not in vocabulary: {unknown}"


_validate_intent_coverage()
