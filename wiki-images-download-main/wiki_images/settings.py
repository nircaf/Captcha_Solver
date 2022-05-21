
BOT_NAME = 'wiki_images'

SPIDER_MODULES = ['wiki_images.spiders']
NEWSPIDER_MODULE = 'wiki_images.spiders'
ITEM_PIPELINES={'scrapy.pipelines.images.ImagesPipeline':1}
# ITEM_PIPELINES = {'wiki_images.pipelines.CustomWikiImagesPipeline': 1}
IMAGES_STORE = 'local_folder'
DOWNLOADER_MIDDLEWARES = {
    'scrapy.downloadermiddlewares.retry.RetryMiddleware': None,
}
ROBOTSTXT_OBEY = True
