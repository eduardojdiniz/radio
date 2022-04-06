Module radio.data.datavisualization
===================================
Data related utilities.

Functions
---------

    
`plot_batch(batch: Dict[str, Any], num_samples: int = 5, random_samples: bool = True, intensities: Optional[List[str]] = None, labels: Optional[List[str]] = None, exclude_keys: Optional[List[str]] = None) ‑> None`
:   plot images and labels from a batch of images

    
`plot_dataset(dataset: torchio.data.dataset.SubjectsDataset, intensities: Optional[List[str]] = None, labels: Optional[List[str]] = None, exclude_keys: Optional[List[str]] = None) ‑> None`
:   plot images and labels from a dataset of subjects

    
`plot_subjects(subjects: List[torchio.data.subject.Subject], num_samples: int = 5, random_samples: bool = True, intensities: Optional[List[str]] = None, labels: Optional[List[str]] = None, exclude_keys: Optional[List[str]] = None) ‑> None`
:   plot images and labels from a batch of images