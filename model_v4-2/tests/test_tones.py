import unittest

from utils import utils

class TestToneHelpers(unittest.TestCase):
    def test_apply_and_detect_tones_on_vowels(self):
        # Vietnamese vowel bases (lower/upper; with diacritic variants)
        bases = [
            'a', 'ă', 'â', 'e', 'ê', 'i', 'o', 'ô', 'ơ', 'u', 'ư', 'y',
            'A', 'Ă', 'Â', 'E', 'Ê', 'I', 'O', 'Ô', 'Ơ', 'U', 'Ư', 'Y',
        ]
        # 0=None, 1=acute, 2=grave, 3=hook, 4=tilde, 5=dot
        for ch in bases:
            for tone_id in range(6):
                out = utils.apply_tone_to_char(ch, tone_id)
                self.assertTrue(utils.is_vietnamese_vowel(out), f"Output not vowel for {ch},{tone_id}: {out}")
                self.assertEqual(utils.tone_of_char(out), tone_id, f"Tone detect mismatch for {ch},{tone_id}: {out}")

            # Applying tone_id=0 should strip tone marks if present
            for tone_id in range(1, 6):
                out = utils.apply_tone_to_char(ch, tone_id)
                back = utils.apply_tone_to_char(out, 0)
                self.assertEqual(back, ch, f"Stripping tone should recover base shape: {ch} -> {out} -> {back}")


if __name__ == '__main__':
    unittest.main()
