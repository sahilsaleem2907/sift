"""App factory: build the core Sift FastAPI application.

Enterprise repos call build_app(), then add their own routers and register
additional forge providers before handing the app to uvicorn.
"""
from fastapi import FastAPI

from sift import config
from sift.integrations.github_client import GitHubClient, github_review_adapter
from sift.integrations.registry import register_forge, register_forge_builder


def build_app() -> FastAPI:
    """Create and configure the core Sift FastAPI app.

    Registers the GitHub forge, mounts all core routers, and runs startup
    side-effects (logging setup, required-env validation).

    Extension points for enterprise callers after this returns:
        app.include_router(...)        # add routes
        register_forge('bitbucket', BitbucketClient)  # add providers
    """
    config.setup_logging()
    config.validate_required()

    register_forge("github", GitHubClient)
    register_forge_builder("github", github_review_adapter)

    from sift.api.feedback import router as feedback_router
    from sift.api.health import router as health_router
    from sift.api.review import router as review_router
    from sift.api.webhooks import router as webhooks_router

    app = FastAPI(title="Sift")
    app.include_router(health_router)
    app.include_router(review_router)
    app.include_router(feedback_router)
    app.include_router(webhooks_router)
    return app
