# src/config.py
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SystemConfig(BaseSettings):
    PRODUCTION: bool = Field(
        default=False,
        description="Flag to indicate if the application is running in production mode",
        alias="PRODUCTION",
    )

    DELETE_COLLECTION: bool = Field(
        default=False,
        description="Flag to indicate if the vector collection should be deleted on shutdown",
        alias="DELETE_COLLECTION",
    )

    MY_COLLECTION_NAME: str = Field(
        default="my_collection",
        description="Name of the vector collection in Qdrant",
        alias="MY_COLLECTION_NAME",
    )

    FOLDER_PATH: str = Field(
        default="./data",
        description="Path to the folder where documents are stored",
        alias="FOLDER_PATH",
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def production(self) -> bool:
        return self.PRODUCTION

    @property
    def delete_collection(self) -> bool:
        return self.DELETE_COLLECTION

    @property
    def collection_name(self) -> str:
        return self.MY_COLLECTION_NAME

    @property
    def folder_path(self) -> str:
        return self.FOLDER_PATH

    @property
    def close_collection_on_shutdown(self) -> bool:
        """
        Determine if the vector collection should be deleted on shutdown.

        Returns:
            bool: True if the collection should be deleted, False otherwise.
        """
        print(
            f"Delete collection: {self.delete_collection}, Production: {self.production}"
        )
        return self.delete_collection and not self.production


@lru_cache(maxsize=1)
def get_settings() -> SystemConfig:
    """Returns a singleton instance of the application configuration."""
    return SystemConfig()
