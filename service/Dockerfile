# The CF planner service (cf-planner.md §2). Build context: the REPOSITORY ROOT, so handler/ can be
# copied beside service/ in the same layout the checkout has.
#
# handler/ is COPIED, never installed from: its requirements.txt carries torch and is never read
# here. The worker's own image builds from ./handler and never sees this file.
FROM python:3.11-slim

# The commit the service runs, a full 40-hex sha (§5). The deploy platform supplies it at build
# time under its own name; unset leaves the service reporting commit null.
ARG RAILWAY_GIT_COMMIT_SHA=
ENV CF_PLANNER_COMMIT=${RAILWAY_GIT_COMMIT_SHA} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY handler/*.py handler/calibration.json /app/handler/
COPY service/ /app/service/
RUN pip install --no-cache-dir -r /app/service/requirements.txt

CMD ["python3", "-m", "service.app"]
