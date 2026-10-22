"""Real offline Chromium checks. Run with system Python (Playwright installed)."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]


class AnnotationBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch(executable_path='/usr/bin/chromium', headless=True,
                                                    args=['--no-sandbox'])

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()

    def setUp(self):
        self.context = self.browser.new_context(accept_downloads=True)
        self.page = self.context.new_page()
        self.errors = []
        self.page.on('pageerror', lambda error: self.errors.append(str(error)))
        self.html = (ROOT / 'data/annotation/pilot-tool.html').read_text()
        self.page.set_content(self.html)

    def tearDown(self):
        self.context.close()
        self.assertEqual(self.errors, [])

    def select_text(self, needle):
        self.page.evaluate('''needle => {
          const node = document.getElementById('source').firstChild;
          const start = node.textContent.indexOf(needle);
          if (start < 0) throw new Error('Fixture passage missing');
          const range = document.createRange();range.setStart(node, start);range.setEnd(node, start+needle.length);
          const selection = window.getSelection();selection.removeAllRanges();selection.addRange(range);
          document.getElementById('source').dispatchEvent(new MouseEvent('mouseup'));
        }''', needle)

    def test_review_export_import_and_source_tampering(self):
        p = self.page
        self.assertEqual(p.locator('#progress').inner_text(), '0 / 20 reviewed')
        p.locator('#jump').select_option('18')
        self.assertIn('mts:train:279', p.locator('#docTitle').inner_text())
        p.locator('#identity').fill('browser-test')
        p.locator('#add').click()
        self.select_text('Micronase')
        p.locator('#useName').click()
        p.locator('#attach').click()
        self.select_text('two point five M G once a day')
        p.locator('#attach').click()
        p.locator('#dose').fill('two point five M G')
        p.locator('#status').select_option('current')
        p.locator('#review').check()
        p.locator('#saveReviewed').click()
        self.assertEqual(p.locator('#stage').inner_text(), 'reviewed')
        with p.expect_download() as event:
            p.locator('#export').click()
        downloaded = json.loads(Path(event.value.path()).read_text())
        doc = downloaded['documents'][18]
        self.assertEqual(doc['medications'][0]['name'], 'Micronase')
        span = doc['medications'][0]['evidence'][0]
        self.assertEqual(doc['text'][span['start']:span['end']], 'Micronase')
        p.goto('about:blank')
        p.set_content(self.html)
        p.locator('#import').set_input_files({'name': 'saved.json', 'mimeType': 'application/json', 'buffer': json.dumps(downloaded).encode()})
        self.assertEqual(p.locator('#progress').inner_text(), '1 / 20 reviewed')
        p.locator('#jump').select_option('18')
        self.assertEqual(p.locator('#name').input_value(), 'Micronase')
        changed = copy.deepcopy(downloaded)
        changed['documents'][18]['text'] += ' changed'
        p.locator('#import').set_input_files({'name': 'changed.json', 'mimeType': 'application/json', 'buffer': json.dumps(changed).encode()})
        self.assertIn('Import failed', p.locator('#message').inner_text())
        self.assertEqual(p.locator('#progress').inner_text(), '1 / 20 reviewed')
        with p.expect_download() as event:
            p.locator('#exportReviewed').click()
        self.assertEqual(len(Path(event.value.path()).read_text().splitlines()), 1)

    def test_empty_result_cannot_be_reviewed_without_confirmation(self):
        p = self.page
        p.locator('#jump').select_option('17')
        p.locator('#identity').fill('browser-test')
        p.locator('#review').check()
        p.locator('#saveReviewed').click()
        self.assertIn('Confirm the empty', p.locator('#message').inner_text())
        p.locator('#negative').check()
        p.locator('#saveReviewed').click()
        self.assertEqual(p.locator('#stage').inner_text(), 'reviewed')
        with p.expect_download() as event:
            p.locator('#export').click()
        saved = json.loads(Path(event.value.path()).read_text())
        self.assertTrue(saved['metadata']['mts:train:109']['negative_confirmed'])

    def test_unicode_selection_and_offline_operation(self):
        seed = json.loads((ROOT / 'data/annotation/pilot-v1.json').read_text())
        seed['documents'] = [dict(id='unicode', text='😀 Take aspirin 5 mg.', medications=[], annotation_status='unannotated')]
        seed['provenance'] = {'unicode': dict(source='fixture', source_split='dev')}
        html = (ROOT / 'annotation/app.html').read_text().replace('__PILOT_DATA__', json.dumps(seed))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'unicode.html'
            path.write_text(html)
            requests = []
            self.page.on('request', lambda request: requests.append(request.url))
            self.page.goto('about:blank')
            self.page.set_content(html)
            self.page.locator('#identity').fill('unicode-test')
            self.page.locator('#add').click()
            self.select_text('aspirin')
            self.page.locator('#useName').click()
            self.page.locator('#attach').click()
            self.page.locator('#saveDraft').click()
            with self.page.expect_download() as event:
                self.page.locator('#export').click()
            saved = json.loads(Path(event.value.path()).read_text())
            self.assertEqual(saved['documents'][0]['medications'][0]['evidence'][0]['start'], 7)
            self.assertTrue(all(url.startswith(('file:', 'blob:')) for url in requests))


if __name__ == '__main__':
    unittest.main()
