# Public Opinion Monitoring

## init

```
uv init public_opinion_monitoring
cd public_opinion_monitoring
uv add dotenv jieba openai pydantic requests spacy tqdm pip setuptools wheel
uv run python -m spacy download zh_core_web_sm   
uv run python -m spacy download en_core_web_sm
```

## spider

```
https://m.weibo.cn/api/container/getIndex?containerid=100103type%3D1%26q%3D食品安全&page_type=searchall&page=1
https://weibo.com/ajax/statuses/mymblog?uid=xxx&page=1

https://github.com/SpiderClub/weibospider
https://github.com/nghuyong/WeiboSpider
```
