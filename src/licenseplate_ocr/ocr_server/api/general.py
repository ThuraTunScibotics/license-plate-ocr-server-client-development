from flask import Flask, Blueprint, render_template, abort

import licenseplate_ocr as lp

class GeneralAPI(Blueprint):

    def __init__(self, **kwargs) -> None:
        super().__init__("general", __name__, **kwargs)
        self.add_url_rule("/", view_func=self.index, methods=["GET"])
        self.add_url_rule("/version", view_func=self.get_version, methods=["GET"])

    def index(self) -> str:
        return render_template('index.html', upload=False)
    
    def get_version(self) -> str:
        try:
            return {"version": str(lp.__version__)}
        except:
            abort(500)