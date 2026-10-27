# Quality Engineering Playbook

## Handling flaky Playwright tests

A flaky test passes and fails intermittently without a product change. Start by reviewing
Playwright traces, screenshots, test video, and the failed step. Avoid adding arbitrary
waitForTimeout delays. Replace timing-based waits with locator auto-waiting and stable
expect assertions. Validate that tests use isolated accounts and reset their own state.

When the same test fails in three consecutive CI runs, open a defect ticket and tag
it as test-flakiness. Assign the owning team, attach the trace, and distinguish
product defects from automation issues. Quarantine a flaky test only with an
owner, a tracking issue, and a removal deadline.

## API test strategy

The API regression suite must validate response status codes, mandatory JSON fields,
contract/schema validation, invalid inputs, authorization, and idempotency where applicable.
Use Playwright APIRequestContext for HTTP API automation. Never place production
credentials or access tokens in the repository.

## Release quality gate

Critical customer checkout flows must pass in the staging environment. There can be
no unresolved severity-1 checkout defect at release time. The release decision
requires a signed-off smoke-test report and an explicit defect-risk summary.
