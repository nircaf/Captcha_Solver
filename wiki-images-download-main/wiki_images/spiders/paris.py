import scrapy



class ParisSpider(scrapy.Spider):
    name = 'paris'
    allowed_domains = []
    # start_urls = ['file:///C:/NeuroBrave/Scrapy/RK-Termin%20-%20Kategorie.html']
    start_urls = ['https://service2.diplo.de/rktermin/extern/appointment_showMonth.do?locationCode=tela&realmId=162&categoryId=271&dateStr=24.07.2022']


    def parse(self, response):
        # raw_image_urls = response.css('.image img ::attr(src)').getall()
        # raw_image_urls = response.xpath("//*[contains(text(), 'data:image/jpg;base64')]").getall()
        raw_image_urls = response.xpath("//*[text()]").getall()
        print("raw_image_urls type " , type(raw_image_urls))

        print("raw_image_urls " , raw_image_urls)
        textfile = open("alltext.txt", "w")
        for element in raw_image_urls:
            textfile.write(element + "\n")
        textfile.close()
        print("done")



