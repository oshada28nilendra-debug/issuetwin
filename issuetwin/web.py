"""Local review server. No issue text is sent to an inference service."""
from urllib.parse import urlsplit
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge
from .service import analyze
from .core import MAX_BYTES


def create_app():
    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=MAX_BYTES + 128 * 1024,
                      TRUSTED_HOSTS=['localhost', '127.0.0.1', '[::1]'])

    @app.before_request
    def guard():
        if request.method == 'POST':
            if request.headers.get('X-IssueTwin') != '1':
                return jsonify(error='Use the local IssueTwin interface.'), 403
            origin = request.headers.get('Origin')
            if origin and urlsplit(origin).netloc != request.host:
                return jsonify(error='Cross-origin requests are not allowed.'), 403

    @app.after_request
    def headers(response):
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        return response

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        return jsonify(error='Export exceeds the 5 MiB limit.'), 413

    @app.get('/')
    def index():
        return render_template('index.html')

    @app.post('/api/analyze')
    def analyze_route():
        upload = request.files.get('file')
        if upload is None:
            return jsonify(error='Choose a JSON or CSV export.'), 400
        try:
            result = analyze(upload.read(MAX_BYTES + 1), upload.filename or '',
                             request.form.get('method', 'minilm'),
                             int(request.form.get('top_k', '30')),
                             float(request.form.get('min_score', '-1')))
            return jsonify(result)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        except RuntimeError as exc:
            return jsonify(error=str(exc)), 503
        except Exception:
            app.logger.exception('Analysis failed')
            return jsonify(error='Analysis failed. Check the server console for details.'), 500

    return app
