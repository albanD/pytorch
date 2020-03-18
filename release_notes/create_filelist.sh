# BASE_VERSION=a5272cb643
BASE_VERSION="tags/v1.5.0"
NEW_VERSION=HEAD

MERGE_BASE=`git merge-base $BASE_VERSION $NEW_VERSION`
echo $MERGE_BASE

git log --reverse --oneline ${MERGE_BASE}..${NEW_VERSION} | cut -d " " -f 1 > commit_list.txt
