# Deployment guide

## Environments
There are three environments: dev, staging and production. Every merge to main deploys to staging automatically.

## Releasing to production
Production releases happen through a manual approval in the CD pipeline. A release needs a green staging smoke test and one approver from the owning team.

## Rollbacks
To roll back, re-run the previous successful pipeline. Rollbacks must be announced in #incidents with the release id.
