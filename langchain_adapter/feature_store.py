# https://github.com/feast-dev/feast
from feast import FeatureStore

from langchain.prompts import PromptTemplate, StringPromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import init_chat_model

from .option import LangChainOption


prompt = PromptTemplate.from_template("""Given the driver's up to date stats, write them note relaying those stats to them.
If they have a conversation rate above .5, give them a compliment. Otherwise, make a silly joke about chickens at the end to make them feel better

Here are the drivers stats:
Conversation rate: {conv_rate}
Acceptance rate: {acc_rate}
Average Daily Trips: {avg_daily_trips}

Your response:""")


store = FeatureStore(repo_path="./langchain_adapter/feast_repo/test/feature_repo")


class FeastPromptTemplate(StringPromptTemplate):
    def format(self, **kwargs) -> str:
        driver_id = kwargs.pop("driver_id")
        feature_vector = store.get_online_features(
            features=[
                "driver_hourly_stats:conv_rate",
                "driver_hourly_stats:acc_rate",
                "driver_hourly_stats:avg_daily_trips",
            ],
            entity_rows=[{"driver_id": driver_id}],
        ).to_dict()
        kwargs["conv_rate"] = feature_vector["conv_rate"][0]
        kwargs["acc_rate"] = feature_vector["acc_rate"][0]
        kwargs["avg_daily_trips"] = feature_vector["avg_daily_trips"][0]
        return prompt.format(**kwargs)


def connecting_to_a_feature_store(config: LangChainOption):
    prompt_template = FeastPromptTemplate(input_variables=["driver_id"])
    model = init_chat_model(
        config.lang_google_chat_model,
        model_provider="google_genai",
        api_key=config.lang_google_api_key,
    )

    return LLMChain(llm=model, prompt=prompt_template)


if __name__ == "__main__":
    chain = connecting_to_a_feature_store(LangChainOption())
    result = chain.invoke(1001)
    print(result)
