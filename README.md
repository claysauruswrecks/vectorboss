# VectorBoss

Operationalized Vector Stores and Datasources via Django, Celery, MySQL.

THE BADDEST DATA HUSTLER IN THE SOFTWARE ENGINEERING HOOD

VectorBoss is about embedding dat AST space into a straight thuggin' high-dimensional vector encoding.

The illest data juggernaut. This street-smart component ingests and vectorates the data so you can sip on that sizzurp instead of sweatin' on those tech docs all day.

It's the ghetto blaster of semantic synthesis, automatic no-slip zone.

## Features

- [ ] API
- [ ] Periodic updates/scraping from datasources
- [ ] Persistent and versioned storage to avoid paying for the same embedding many times (git repos have many commits)
  - [ ] Only allow for latest commit on a repository source
- [ ] Find/audit source data from vector search results
- [ ] Django admin for managing versions of llamahub sources
  - [ ] Source data inspection via admin
  - [ ] Source data deletion via admin
  - [ ] Source data migration via admin
- [ ] Audits
  - [ ] 5 W's attached to loader instances through time
    - Who, what, when, where, why
  - [ ] Rollback to a specific point in time

## API

> TODO: Refactor this design

- [ ] /load - loads the data from MySQL
- [ ] /status - healthcheck
- [ ] /update - scrape for new information
- [ ] /check - returns inconsistencies between docs and the code
