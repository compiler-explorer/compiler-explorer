# This document is a Work in Progress

This document pretends to put everything that is important to check before doing a full site release as well as to how
to do it, to ensure that the site fully works upon the release of a new version.

Depending on the nature of the changes, only some sections of this list will be relevant - Feel free to focus only on
those you deem most important.

If you think we are missing any important point, we greatly appreciate suggestions.

## Running the site

First things first, we'll use the beta environment as it lets us run the site under additional checks that will automate
some of the problem detections for us.

To run the new version on the beta environment we'll use the `ce` utility:

Ensure that the beta environment is active by running `ce --env beta environment start` - You can skip this step if it's
already running from a previous run or if you'd rather boot it once the current version is marked as active.

Check what `versionId` the desired commit has. This can be checked on the Github CI build logs as its build number for
that specific commit, or by running `ce --env beta builds list` which shows a list of the most recent builds with their
corresponding commit hash.

Once you have the desired `versionId`, mark it as the current version to be used by running
`ce --env beta builds set_current {versionId}`

If needed, you can now restart the currently running instances with `ce --env beta instances restart` if needed.

Once this is finished, the `/beta` endpoint should be ready for testing.

- The first issue you might find is that the beta instance does not boot. This might be caused by the
  `--ensureNoIdClash` flag shutting the app down if it detects one or more pairs of compilers sharing the same id even
  if they belong to different languages. An error should be logged with the culprits for easy debugging.

## Basic general site functionality

In-depth documentation follows below, but once the beta site is running, some basic checks include:

-
