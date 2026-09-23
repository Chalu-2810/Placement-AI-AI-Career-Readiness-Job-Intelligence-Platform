"""
scripts/refresh_job_market.py
-------------------------------
Refreshes the Job Market Intelligence snapshots (Phase 5).

Run manually or on a schedule (cron), OUTSIDE the running web app:

    python scripts/refresh_job_market.py

This is deliberately NOT a Flask route -- there is no way for a student
or an unauthenticated visitor to trigger a RemoteOK fetch on demand. The
running app (app.py) only ever reads the latest already-saved snapshot
from job_market_snapshots via database.get_latest_job_market_snapshot().

What this script does, per the approved architecture:

    RemoteOK (one HTTP call)
        -> classify postings into the app's 8 existing domains
        -> extract skills per posting (utils/resume.py's existing taxonomy)
        -> aggregate into one snapshot dict per domain
        -> INSERT one row per domain into job_market_snapshots

Failures (network error, malformed response, DB error) are logged and
this script exits non-zero -- it never writes a partial/fabricated
snapshot on failure.
"""

import sys
import os
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

import database as db
import job_market

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger('placementai.refresh_job_market')


def main():
    logger.info('Starting job market refresh from %s', job_market.REMOTEOK_API_URL)

    try:
        snapshots = job_market.refresh_all_domains()
    except requests.RequestException as e:
        logger.error('RemoteOK request failed: %s', e)
        return 1
    except ValueError as e:
        logger.error('RemoteOK response was malformed: %s', e)
        return 1
    except Exception:
        logger.exception('Unexpected error during job market refresh')
        return 1

    saved = 0
    for domain, snapshot in snapshots.items():
        try:
            db.save_job_market_snapshot(
                domain               = domain,
                source               = snapshot['source'],
                fetched_at           = snapshot['fetched_at'],
                posting_count        = snapshot['posting_count'],
                raw_aggregates_json  = json.dumps(snapshot),
            )
            saved += 1
            logger.info('Saved snapshot: domain=%s postings=%d top_skills=%s',
                        domain, snapshot['posting_count'],
                        [s['skill'] for s in snapshot['skill_frequencies'][:5]])
        except Exception:
            logger.exception('Failed to save snapshot for domain=%s', domain)

    logger.info('Job market refresh complete: %d/%d domain snapshots saved', saved, len(snapshots))
    return 0 if saved == len(snapshots) else 1


if __name__ == '__main__':
    sys.exit(main())
