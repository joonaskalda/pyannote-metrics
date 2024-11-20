#!/usr/bin/env python
# encoding: utf-8

from typing import Union, Optional, List, Dict
from .base import BaseMetric
from meeteval.wer.wer.cp import cp_word_error_rate
import warnings
import os
from whisper.normalizers import EnglishTextNormalizer
from pyannote.core import Annotation
class CpWER(BaseMetric):
    """
    Continuous Phrase Word Error Rate (CpWER) Metric

    CpWER is defined as:
        CpWER = (Substitutions + Deletions + Insertions) / Length

    Attributes
    ----------
    name : str
        Name of the metric ('CpWER')
    components : List[str]
        Components used to compute the metric ('length', 'substitutions', 'deletions', 'insertions')
    reference_dir : str
        Directory containing reference text files.
    """

    def __init__(self, reference_dir: str = "/Users/joonaskalda/data/ami_public_manual_1.6.2/words_joined", **kwargs):
        """
        Initialize the CpWER metric.

        Parameters
        ----------
        reference_dir : str
            Directory containing reference text files.
        """
        super().__init__(**kwargs)
        self.reference_dir = reference_dir
        self.normalizer = EnglishTextNormalizer()

    @classmethod
    def metric_name(cls) -> str:
        return "CpWER"

    @classmethod
    def metric_components(cls) -> List[str]:
        return ["length", "substitutions", "deletions", "insertions"]

    def compute_components(
        self,
        reference: Annotation,
        hypothesis: Union[List[str], str],
        **kwargs
    ) -> Dict[str, float]:
        """
        Compute the CpWER components for a given sample.

        Parameters
        ----------
        reference : Annotation
            Reference annotation.
        hypothesis : List[str] or str
            The predicted transcription(s).

        Returns
        -------
        components : Dict[str, float]
            Dictionary containing 'length', 'substitutions', 'deletions', 'insertions'.
        """
        # hypothesis should be a tuple with 3 elements (diarization, asr_hypothesis, sources)
        asr_hypothesis = hypothesis[1]

        uri = reference._uri
        references_file_paths = [
            os.path.join(self.reference_dir, file) for file in os.listdir(self.reference_dir) if file.startswith(uri)
        ]
        reference_text = []
        for reference_file_path in references_file_paths:
            with open(reference_file_path, 'r', encoding='utf-8') as ref_file:
                reference_text.append(self.normalizer(ref_file.read().strip()))

        # Ensure that reference and hypothesis are lists of strings
        if isinstance(reference_text, str):
            reference_text = [reference_text]
        else:
            reference_text = reference_text

        if isinstance(asr_hypothesis, str):
            asr_hypothesis = [asr_hypothesis]
        else:
            asr_hypothesis = asr_hypothesis

        asr_hypothesis = [self.normalizer(hyp) for hyp in asr_hypothesis]

        total_length = 0.0
        total_subs = 0.0
        total_dels = 0.0
        total_ins = 0.0
        cpwer = cp_word_error_rate(reference_text, asr_hypothesis)
        total_length += cpwer.length
        total_subs += cpwer.substitutions
        total_dels += cpwer.deletions
        total_ins += cpwer.insertions

        return {
            "length": total_length,
            "substitutions": total_subs,
            "deletions": total_dels,
            "insertions": total_ins
        }

    def compute_metric(self, components: Dict[str, float]) -> float:
        """
        Compute the CpWER metric from its components.

        Parameters
        ----------
        components : Dict[str, float]
            Dictionary containing 'length', 'substitutions', 'deletions', 'insertions'.

        Returns
        -------
        cpwer : float
            The computed CpWER value.
        """
        numerator = components["substitutions"] + components["deletions"] + components["insertions"]
        denominator = components["length"]

        if denominator == 0.0:
            if numerator == 0.0:
                return 0.0
            else:
                warnings.warn("Denominator (length) is zero while numerator is non-zero. Returning CpWER as 1.0.")
                return 1.0
        else:
            return numerator / denominator
    
    def __call__(self, reference: Annotation, hypothesis: Union[List[str], str],
                 detailed: bool = False, uri: Optional[str] = None, **kwargs):
        """Compute metric value and accumulate components

        Parameters
        ----------
        reference : type depends on the metric
            Manual `reference`
        hypothesis : type depends on the metric
            Evaluated `hypothesis`
        uri : optional
            Override uri.
        detailed : bool, optional
            By default (False), return metric value only.
            Set `detailed` to True to return dictionary where keys are
            components names and values are component values

        Returns
        -------
        value : float (if `detailed` is False)
            Metric value
        components : dict (if `detailed` is True)
            `components` updated with metric value
        """

        # compute metric components
        components = self.compute_components(reference, hypothesis, **kwargs)

        # compute rate based on components
        components[self.metric_name_] = self.compute_metric(components)

        # keep track of this computation
        uri = reference._uri
        self.results_.append((uri, components))

        # accumulate components
        for name in self.components_:
            self.accumulated_[name] += components[name]

        if detailed:
            return components

        return components[self.metric_name_]
