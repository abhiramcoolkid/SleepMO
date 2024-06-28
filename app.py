from flask import Flask, request, render_template
import mne
import matplotlib.pyplot as plt
import numpy as np
import sys

subject_number = None
session_number = None
name = None
age = None
specificity = None
SleepSchedule = None

#datapath = mne.datasets.sample.data_path() / "MEG" / "Sample"
#subjects_dir = datapath.parents[1] / "subjects"

#mne.viz.Brain(surf="pial", hemi="lh", subject="sample", subjects_dir=subjects_dir)

#<div>
              #      <label for="schedule">Sleep Schedule:</label>
               #     <input type="text" id="schedule" name="schedule" placeholder="Example: 7 hours" required>
                #</div>
                #<br>

                #<div>
                #    <label for="subject">Subject Number:</label>
                #    <input type="number" id="subject" name="subject" placeholder="Subject Number" required min="0" max="2">
                #</div>
#<div>
 #                   <label for="schedule">Sleep Schedule:</label>
  #                  <input type="text" id="schedule" name="schedule" placeholder="Example: 7 hours" required>
   #             </div>
    #            <br>

#                <div>
 #                   <label for="subject">Subject Number:</label>
  #                  <input type="number" id="subject" name="subject" placeholder="Subject Number" required min="0" max="2">
   #             </div>

    #            <br>

                #<br>

from waitress import serve

data_path = mne.datasets.ssvep.data_path()
data_path2 = mne.datasets.sample.data_path()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():

    global subject_number, session_number, specificity, name, age, SleepSchedule, dataform7
    data_from_form = request.form['input_name']
    dataform5 = request.form['age']
    dataform4 = request.form['title']
    dataform6 = request.form['schedule']
    dataform7 = request.form['condition']

    name = data_from_form
    age = int(dataform5)

    specificity = dataform4
    SleepSchedule=dataform6
    #subject_number = 23
    #session_number = 12

    if dataform7 == "Sleepy" or dataform7 == "sleepy":
        subject_number = "2"
        session_number = "1"
    elif dataform7 == "Good" or dataform7 == "good":
        subject_number ="1"
        session_number = "1"
    else:
        pass

    bird_fname = data_path / f"sub-0{subject_number}" / f"ses-0{session_number}" / "eeg" / f"sub-0{subject_number}_ses-0{session_number}_task-ssvep_eeg.vhdr"

    raw = mne.io.read_raw_brainvision(bird_fname, verbose=False, preload=True)
    raw.info["line_freq"] = 50.0

    montage = mne.channels.make_standard_montage("easycap-M1")
    raw.set_montage(montage, verbose=False)

    raw.set_eeg_reference("average", projection=False, verbose=False)

    raw.filter(l_freq=0.1, h_freq=None, fir_design="firwin", verbose=False)

    event_id = {"12hz": 255, "15hz": 155}
    events, _ = mne.events_from_annotations(raw, verbose=False)
    tmin, tmax = -0.1, 20.0
    baseline = None
    epochs = mne.Epochs(raw, events=events, baseline=baseline, verbose=False, tmax=tmax, tmin=tmin,
                        event_id=[event_id["12hz"], event_id["15hz"]], )

    tmin = 1.0
    tmax = 20.0
    fmin = 1.0
    fmax = 90.0
    sfreq = epochs.info["sfreq"]

    spectrum = epochs.compute_psd(
        "welch",
        n_fft=int(sfreq * (tmax - tmin)),
        n_overlap=0,
        n_per_seg=None,
        tmin=tmin,
        tmax=tmax,
        fmin=fmin,
        fmax=fmax,
        window="boxcar",
        verbose=False,
    )
    psds, freqs = spectrum.get_data(return_freqs=True)

    def snr_spectrum(psd, noise_n_neighbor_freqs=1, noise_skip_neighbor_freqs=1):
        average_kernel = np.concatenate(
            (
                np.ones(noise_n_neighbor_freqs),
                np.zeros(2 * noise_skip_neighbor_freqs + 1),
                np.ones(noise_n_neighbor_freqs),

            )
        )
        average_kernel /= average_kernel.sum()
        mean_noise = np.apply_along_axis(
            lambda psd_: np.convolve(psd_, average_kernel, mode="valid"), axis=-1, arr=psd

        )

        edge_width = noise_n_neighbor_freqs + noise_skip_neighbor_freqs
        pad_width = [(0, 0)] * (mean_noise.ndim - 1) + [(edge_width, edge_width)]
        mean_noise = np.pad(mean_noise, pad_width=pad_width, constant_values=np.nan)

        return psd / mean_noise

    snrs = snr_spectrum(psds, noise_n_neighbor_freqs=3, noise_skip_neighbor_freqs=1)

    # Graph

    #fig, axes = plt.subplots(2, 1, sharex="all", sharey="none", figsize=(8, 5))
    #freq_range = range(
    #    np.where(np.floor(freqs) == 1.0)[0][0], np.where(np.ceil(freqs) == fmax - 1)[0][0]
    #)

    #psds_plot = 10 * np.log10(psds)
    #psds_mean = psds_plot.mean(axis=(0, 1))[freq_range]
    #psds_std = psds_plot.std(axis=(0, 1))[freq_range]
    #axes[0].plot(freqs[freq_range], psds_mean, color="blue")
    #axes[0].fill_between(
    #    freqs[freq_range], psds_mean - psds_std, psds_mean + psds_std, alpha=0.2, color="blue"
    #)
    #axes[0].set(title="PSD Spectrum", ylabel="Power Spectral Density [dB]")

    #snr_mean = psds_plot.mean(axis=(0, 1))[freq_range]
    #snr_std = psds_plot.std(axis=(0, 1))[freq_range]
    #axes[1].plot(freqs[freq_range], snr_mean, color="red")
    #axes[1].fill_between(
    #    freqs[freq_range], snr_mean - snr_std, snr_mean + snr_std, alpha=0.2, color="red"
    #)
    #axes[1].set(title="SNR Spectrum", xlabel="Frequency [dB]", ylabel="Signal-To-Noise Ratio", ylim=[-2, 30],
    #            xlim=[fmin, fmax])
    #fig.show()

    stim_freq = 12.0

    i_bin_12hz = np.argmin(abs(freqs - stim_freq))

    i_bin_24hz = np.argmin(abs(freqs - 24))
    i_bin_36hz = np.argmin(abs(freqs - 36))
    i_bin_15hz = np.argmin(abs(freqs - 15))
    i_bin_30hz = np.argmin(abs(freqs - 30))
    i_bin_45hz = np.argmin(abs(freqs - 45))

    i_identify_trial_12hz = np.where(epochs.events[:, 2] == event_id["12hz"])[0]
    i_identify_trial_15hz = np.where(epochs.events[:, 2] == event_id["15hz"])[0]

    roi_vis = [
        "POz",
        "Oz",
        "O1",
        "O2",
        "PO3",
        "PO4",
        "PO7",
        "PO8",
        "PO9",
        "PO10",
        "O9",
        "O10",
    ]  # visual roi

    picks_from_roi_viz = mne.pick_types(
        epochs.info, exclude="bads", eeg=True, stim=False, selection=roi_vis,
    )

    snrs_roi = snrs[i_identify_trial_12hz, :, i_bin_12hz][:, picks_from_roi_viz]
    average_snr = int(snrs_roi.mean())
    print(f"Subject 2: SNR Data from 12hz trial")
    value_szn = raw.info["bads"]
    print(f"Average SNR (ROI): {snrs_roi.mean()}")
    print(f"Rounded version: {average_snr}")

    SNR_predict_melatonin = {
        41: "75",
        25: "43",
        11: "25",
        15: "30"

    }

    melatonin_get = SNR_predict_melatonin.get(average_snr, " ")


    bird_fname2 = data_path2 / "MEG" / "Sample" / "sample_audvis_raw.fif"

    #new_raw = mne.io.read_raw_fif(bird_fname2, preload=True, verbose=False)
    report = mne.Report(title=name + "'s Data Analysis")
    #new_raw.pick(picks=["eeg"]).crop(tmax=100).load_data()
    #report.add_raw(raw=new_raw, title="Report")
    report.save("report_raw.html", overwrite=True)
    print(f"We are generating {name}'s data report")

    if melatonin_get == "75":
        myhtml = f"""
        <h1> {name}'s Personalized Sleep Recommendations </h1>
        <p>
        Through computed <b>melatonin</b> levels, here's what we suggest you do:
        <p></p>
        <p>
        1. Make sure to stay hydrated
        </p>
        <p></p>
        <p>
        2. Make sure to limit the exposure to blue light before you sleep
        </p>
        <p></p>
        <p>
        3. Make sure to drop your body temperature before you sleep
        </p>
        """
        report.add_html(html=myhtml, title=name + "'s Personalized Recommendations")
        report.save("report_add_html.html", overwrite=True)
        if SleepSchedule == "10 hours":
            if age < 18:
                myhtml = f"""
                <h1> Great Job {name}! </h1>
                <p> This will help with sleep immensely. Stay consistent and continue to endure growth and development. </p>
                """
                report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                else:
                    pass

            elif age == 18:
                myhtml = """
                    <h1> Great Job </h1>
                    <p> This will help with sleep immensely. Stay consistent and continue to endure growth and development. </p>
                    """
                report.add_html(html=myhtml, title="data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > 18 and age < 27:
                myhtml = """
                <h1> Keep Doing What You're Doing </h1>
                <p> 10 hours is probably a little too much, but it helps with growth. </p>
                <p></p>
                <p> As you'll be wrapping up your growth within the ages of 18-27, this is the time where you shouldn't compromise your growth. Keep doing what you're doing. </p>
                """
                report.add_html(html=myhtml, title="data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > 30 and age < 45:
                myhtml = """
                <h1> Wow </h1>
                <p> That's a great amount of sleep. Great job!
                """
                report.add_html(html=myhtml, title="data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > "55" and age < "70":
                myhtml = """
                    <h1> At your age, the main goal would be to aim for more sleep </h1>
                    """
                report.add_html(html=myhtml, title="data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
        elif SleepSchedule == "9 hours":
            if age == 18:
                myhtml = """
                    <h1> Good Job </h1>
                    <p> This will help with sleep immensely. Stay consistent and continue to endure growth and development. </p>
                    """
                report.add_html(html=myhtml, title="data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > "18" and age < "27":
                myhtml = """
                <h1> Keep Doing What You're Doing </h1>
                <p> 9 hours is a great amount of sleep </p>
                <p></p>
                <p> As you'll be wrapping up your growth within the ages of 18-27, this is the time where you shouldn't compromise your growth. Keep doing what you're doing. </p>
                """
                report.add_html(html=myhtml, title="data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > "30" and age < "45":
                myhtml = """
                <h1> Wow </h1>
                <p> That's a good amount of sleep. Great job!
                """
                report.add_html(html=myhtml, title="data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > "55" and age < "70":
                myhtml = """
                    <h1> At your age, the main goal would be to aim for more sleep </h1>
                    """
                report.add_html(html=myhtml, title="data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
        elif SleepSchedule == "8 hours":
            if age < 18:
                myhtml = f"""
                <h1> 8 hours! </h1>
                <p> Great job <b>{name}</b>! Continue to do what you are doing. Stick to this sleep schedule and you will be great. </p>"""
                report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age == 18:
                myhtml = f"""
                <h1> Sleep Recommendations </h1>
                <p> You're still growing {name}, so make sure to stay consistent and endure the growth.</p>
                """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > 18 and age < 27:
                myhtml = f"""
                <h1> {name}'s Sleep Recommendations </h1>
                <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
                """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > 30 and age < 45:
                myhtml = f"""
                <h1> {name}'s Sleep Recommendations </h1>
                <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
                """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > "55" and age < "70":
                myhtml = """
                <h1> At your age, the main goal would be to aim for more sleep </h1>
                """
                report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
        elif SleepSchedule == "7 hours":
            if age < 18:
                myhtml = f"""
                <h1> 8 hours! </h1>
                <p> Great job <b>{name}</b>! Continue to do what you are doing. Stick to this sleep schedule and you will be great. </p>"""
                report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age == 18:
                myhtml = f"""
                <h1> Sleep Recommendations </h1>
                <p> You're still growing {name}, so make sure to stay consistent and endure the growth.</p>
                """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > 18 and age < 27:
                myhtml = f"""
                <h1> {name}'s Sleep Recommendations </h1>
                <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
                """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > 30 and age < 45:
                myhtml = f"""
                <h1> {name}'s Sleep Recommendations </h1>
                <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
                """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > "55" and age < "70":
                myhtml = """
                <h1> At your age, the main goal would be to aim for more sleep </h1>
                """
                report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)

        elif SleepSchedule == "6 hours":
            if age < 18:
                myhtml = f"""
                <h1> 8 hours! </h1>
                <p> Great job <b>{name}</b>! Continue to do what you are doing. Stick to this sleep schedule and you will be great. </p>"""
                report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age == 18:
                myhtml = f"""
                <h1> Sleep Recommendations </h1>
                <p> You're still growing {name}, so make sure to stay consistent and endure the growth.</p>
                """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > 18 and age < 27:
                myhtml = f"""
                <h1> {name}'s Sleep Recommendations </h1>
                <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
                """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > 30 and age < 45:
                myhtml = f"""
                <h1> {name}'s Sleep Recommendations </h1>
                <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
                """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > "55" and age < "70":
                myhtml = """
                <h1> At your age, the main goal would be to aim for more sleep </h1>
                """
                report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
        elif SleepSchedule == "5 hours":
            if age < 18:
                myhtml = f"""
                <h1> 8 hours! </h1>
                <p> Great job <b>{name}</b>! Continue to do what you are doing. Stick to this sleep schedule and you will be great. </p>"""
                report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age == 18:
                myhtml = f"""
                <h1> Sleep Recommendations </h1>
                <p> You're still growing {name}, so make sure to stay consistent and endure the growth.</p>
                """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > 18 and age < 27:
                myhtml = f"""
                <h1> {name}'s Sleep Recommendations </h1>
                <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
                """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > 30 and age < 45:
                myhtml = f"""
                <h1> {name}'s Sleep Recommendations </h1>
                <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
                """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > "55" and age < "70":
                myhtml = """
                <h1> At your age, the main goal would be to aim for more sleep </h1>
                """
                report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)

        elif SleepSchedule == "11 hours":
            if age < 18:
                myhtml = f"""
                <h1> 8 hours! </h1>
                <p> Great job <b>{name}</b>! Continue to do what you are doing. Stick to this sleep schedule and you will be great. </p>"""
                report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age == 18:
                myhtml = f"""
                <h1> Sleep Recommendations </h1>
                <p> You're still growing {name}, so make sure to stay consistent and endure the growth.</p>
                """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > 18 and age < 27:
                myhtml = f"""
                <h1> {name}'s Sleep Recommendations </h1>
                <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
                """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > 30 and age < 45:
                myhtml = f"""
                <h1> {name}'s Sleep Recommendations </h1>
                <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
                """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > "55" and age < "70":
                myhtml = """
                <h1> At your age, the main goal would be to aim for more sleep </h1>
                """
                report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
    elif melatonin_get == "45":
        myhtml = f"""
                <h1> {name}'s Personalized Sleep Recommendations </h1>
                <p>
                Through computed <b>melatonin</b> levels, here's what we suggest you do:
                <p></p>
                <p>
                1. Make sure to stay hydrated
                </p>
                <p></p>
                <p>
                2. Make sure to limit the exposure to blue light before you sleep
                </p>
                <p></p>
                <p>
                3. Make sure to drop your body temperature before you sleep
                </p>
                """
        report.add_html(html=myhtml, title=name + "'s Personalized Recommendations")
        report.save("report_add_html.html", overwrite=True)
        if SleepSchedule == "10 hours":
            if age < 18:
                myhtml = f"""
                    <h1> Great Job {name}! </h1>
                    <p> This will help with sleep immensely. Stay consistent and continue to endure growth and development. </p>
                    """
                report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age == 18:
                myhtml = """
                        <h1> Great Job </h1>
                        <p> This will help with sleep immensely. Stay consistent and continue to endure growth and development. </p>
                        """
                report.add_html(html=myhtml, title="data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > 18 and age < 27:
                myhtml = """
                    <h1> Keep Doing What You're Doing </h1>
                    <p> 10 hours is probably a little too much, but it helps with growth. </p>
                    <p></p>
                    <p> As you'll be wrapping up your growth within the ages of 18-27, this is the time where you shouldn't compromise your growth. Keep doing what you're doing. </p>
                    """
                report.add_html(html=myhtml, title="data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > 30 and age < 45:
                myhtml = """
                    <h1> Wow </h1>
                    <p> That's a great amount of sleep. Great job!
                    """
                report.add_html(html=myhtml, title="data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > "55" and age < "70":
                myhtml = """
                        <h1> At your age, the main goal would be to aim for more sleep </h1>
                        """
                report.add_html(html=myhtml, title="data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
        elif SleepSchedule == "9 hours":
            if age == 18:
                myhtml = """
                        <h1> Good Job </h1>
                        <p> This will help with sleep immensely. Stay consistent and continue to endure growth and development. </p>
                        """
                report.add_html(html=myhtml, title="data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > "18" and age < "27":
                myhtml = """
                    <h1> Keep Doing What You're Doing </h1>
                    <p> 9 hours is a great amount of sleep </p>
                    <p></p>
                    <p> As you'll be wrapping up your growth within the ages of 18-27, this is the time where you shouldn't compromise your growth. Keep doing what you're doing. </p>
                    """
                report.add_html(html=myhtml, title="data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > "30" and age < "45":
                myhtml = """
                    <h1> Wow </h1>
                    <p> That's a good amount of sleep. Great job!
                    """
                report.add_html(html=myhtml, title="data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > "55" and age < "70":
                myhtml = """
                        <h1> At your age, the main goal would be to aim for more sleep </h1>
                        """
                report.add_html(html=myhtml, title="data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
        elif SleepSchedule == "8 hours":
            if age < 18:
                myhtml = f"""
                    <h1> 8 hours! </h1>
                    <p> Great job <b>{name}</b>! Continue to do what you are doing. Stick to this sleep schedule and you will be great. </p>"""
                report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age == 18:
                myhtml = f"""
                    <h1> Sleep Recommendations </h1>
                    <p> You're still growing {name}, so make sure to stay consistent and endure the growth.</p>
                    """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > 18 and age < 27:
                myhtml = f"""
                    <h1> {name}'s Sleep Recommendations </h1>
                    <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
                    """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > 30 and age < 45:
                myhtml = f"""
                    <h1> {name}'s Sleep Recommendations </h1>
                    <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
                    """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > "55" and age < "70":
                myhtml = """
                    <h1> At your age, the main goal would be to aim for more sleep </h1>
                    """
                report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
        elif SleepSchedule == "7 hours":
            if age < 18:
                myhtml = f"""
                    <h1> 8 hours! </h1>
                    <p> Great job <b>{name}</b>! Continue to do what you are doing. Stick to this sleep schedule and you will be great. </p>"""
                report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age == 18:
                myhtml = f"""
                    <h1> Sleep Recommendations </h1>
                    <p> You're still growing {name}, so make sure to stay consistent and endure the growth.</p>
                    """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > 18 and age < 27:
                myhtml = f"""
                    <h1> {name}'s Sleep Recommendations </h1>
                    <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
                    """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > 30 and age < 45:
                myhtml = f"""
                    <h1> {name}'s Sleep Recommendations </h1>
                    <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
                    """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > "55" and age < "70":
                myhtml = """
                    <h1> At your age, the main goal would be to aim for more sleep </h1>
                    """
                report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)

        elif SleepSchedule == "6 hours":
            if age < 18:
                myhtml = f"""
                    <h1> 8 hours! </h1>
                    <p> Great job <b>{name}</b>! Continue to do what you are doing. Stick to this sleep schedule and you will be great. </p>"""
                report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age == 18:
                myhtml = f"""
                    <h1> Sleep Recommendations </h1>
                    <p> You're still growing {name}, so make sure to stay consistent and endure the growth.</p>
                    """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > 18 and age < 27:
                myhtml = f"""
                    <h1> {name}'s Sleep Recommendations </h1>
                    <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
                    """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > 30 and age < 45:
                myhtml = f"""
                    <h1> {name}'s Sleep Recommendations </h1>
                    <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
                    """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > "55" and age < "70":
                myhtml = """
                    <h1> At your age, the main goal would be to aim for more sleep </h1>
                    """
                report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
        elif SleepSchedule == "5 hours":
            if age < 18:
                myhtml = f"""
                    <h1> 8 hours! </h1>
                    <p> Great job <b>{name}</b>! Continue to do what you are doing. Stick to this sleep schedule and you will be great. </p>"""
                report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age == 18:
                myhtml = f"""
                    <h1> Sleep Recommendations </h1>
                    <p> You're still growing {name}, so make sure to stay consistent and endure the growth.</p>
                    """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > 18 and age < 27:
                myhtml = f"""
                    <h1> {name}'s Sleep Recommendations </h1>
                    <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
                    """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > 30 and age < 45:
                myhtml = f"""
                    <h1> {name}'s Sleep Recommendations </h1>
                    <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
                    """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""
    elif melatonin_get == "25":
        myhtml = f"""
            <h1> {name}'s Personalized Sleep Recommendations </h1>
            <p>
            Through computed <b>melatonin</b> levels, here's what we suggest you do:
            <p></p>
            <p>
            1. Make sure to stay hydrated
            </p>
            <p></p>
            <p>
            2. Make sure to limit the exposure to blue light before you sleep
            </p>
            <p></p>
            <p>
            3. Make sure to drop your body temperature before you sleep
            </p>
            """
        report.add_html(html=myhtml, title=name + "'s Personalized Recommendations")
        report.save("report_add_html.html", overwrite=True)
        if SleepSchedule == "10 hours":
            if age < 18:
                myhtml = f"""
                <h1> Great Job {name}! </h1>
                <p> This will help with sleep immensely. Stay consistent and continue to endure growth and development. </p>
                """
                report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age == 18:
                myhtml = """
                    <h1> Great Job </h1>
                    <p> This will help with sleep immensely. Stay consistent and continue to endure growth and development. </p>
                    """
                report.add_html(html=myhtml, title="data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > 18 and age < 27:
                myhtml = """
                <h1> Keep Doing What You're Doing </h1>
                <p> 10 hours is probably a little too much, but it helps with growth. </p>
                <p></p>
                <p> As you'll be wrapping up your growth within the ages of 18-27, this is the time where you shouldn't compromise your growth. Keep doing what you're doing. </p>
                """
                report.add_html(html=myhtml, title="data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General ways to improve sleep")
                    report.save("report_add_html.html", overwrite=True)
            elif age > 30 and age < 45:
                myhtml = """
                <h1> Wow </h1>
                <p> That's a great amount of sleep. Great job!
                """
                report.add_html(html=myhtml, title="data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General ways to improve sleep")
                    report.save("report_add_html.html", overwrite=True)
            elif age > "55" and age < "70":
                myhtml = """
                    <h1> At your age, the main goal would be to aim for more sleep </h1>
                    """
                report.add_html(html=myhtml, title="data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General ways to improve sleep")
                    report.save("report_add_html.html", overwrite=True)
        elif SleepSchedule == "9 hours":
            if age == 18:
                myhtml = """
                    <h1> Good Job </h1>
                    <p> This will help with sleep immensely. Stay consistent and continue to endure growth and development. </p>
                    """
                report.add_html(html=myhtml, title="data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General ways to improve sleep")
                    report.save("report_add_html.html", overwrite=True)
            elif age > "18" and age < "27":
                myhtml = """
                <h1> Keep Doing What You're Doing </h1>
                <p> 9 hours is a great amount of sleep </p>
                <p></p>
                <p> As you'll be wrapping up your growth within the ages of 18-27, this is the time where you shouldn't compromise your growth. Keep doing what you're doing. </p>
                """
                report.add_html(html=myhtml, title="data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General ways to improve sleep")
                    report.save("report_add_html.html", overwrite=True)
            elif age > "30" and age < "45":
                myhtml = """
                <h1> Wow </h1>
                <p> That's a good amount of sleep. Great job!
                """
                report.add_html(html=myhtml, title="data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General ways to improve sleep")
                    report.save("report_add_html.html", overwrite=True)
            elif age > "55" and age < "70":
                myhtml = """
                    <h1> At your age, the main goal would be to aim for more sleep </h1>
                    """
                report.add_html(html=myhtml, title="data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General ways to improve sleep")
                    report.save("report_add_html.html", overwrite=True)
        elif SleepSchedule == "8 hours":
            if age < 18:
                myhtml = f"""
                <h1> 8 hours! </h1>
                <p> Great job <b>{name}</b>! Continue to do what you are doing. Stick to this sleep schedule and you will be great. </p>"""
                report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General ways to improve sleep")
                    report.save("report_add_html.html", overwrite=True)
            elif age == 18:
                myhtml = f"""
                <h1> Sleep Recommendations </h1>
                <p> You're still growing {name}, so make sure to stay consistent and endure the growth.</p>
                """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General ways to improve sleep")
                    report.save("report_add_html.html", overwrite=True)
            elif age > 18 and age < 27:
                myhtml = f"""
                <h1> {name}'s Sleep Recommendations </h1>
                <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
                """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General ways to improve sleep")
                    report.save("report_add_html.html", overwrite=True)
            elif age > 30 and age < 45:
                myhtml = f"""
                <h1> {name}'s Sleep Recommendations </h1>
                <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
                """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General ways to improve sleep")
                    report.save("report_add_html.html", overwrite=True)
            elif age > "55" and age < "70":
                myhtml = """
                <h1> At your age, the main goal would be to aim for more sleep </h1>
                """
                report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General ways to improve sleep")
                    report.save("report_add_html.html", overwrite=True)
        elif SleepSchedule == "7 hours":
            if age < 18:
                myhtml = f"""
                <h1> 8 hours! </h1>
                <p> Great job <b>{name}</b>! Continue to do what you are doing. Stick to this sleep schedule and you will be great. </p>"""
                report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com</b>. </i> Find this code on <a href="www.github.com/AbhiramRuthala">  GitHub! </a> </p>"""

                    report.add_html(html=myhtml, title="General ways to improve sleep")
                    report.save("report_add_html.html", overwrite=True)
            elif age == 18:
                myhtml = f"""
                <h1> Sleep Recommendations </h1>
                <p> You're still growing {name}, so make sure to stay consistent and endure the growth.</p>
                """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General ways to improve sleep")
                    report.save("report_add_html.html", overwrite=True)
            elif age > 18 and age < 27:
                myhtml = f"""
                <h1> {name}'s Sleep Recommendations </h1>
                <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
                """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General ways to improve sleep")
                    report.save("report_add_html.html", overwrite=True)
            elif age > 30 and age < 45:
                myhtml = f"""
                <h1> {name}'s Sleep Recommendations </h1>
                <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
                """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General ways to improve sleep")
                    report.save("report_add_html.html", overwrite=True)
            elif age > "55" and age < "70":
                myhtml = """
                <h1> At your age, the main goal would be to aim for more sleep </h1>
                """
                report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General ways to improve sleep")
                    report.save("report_add_html.html", overwrite=True)

        elif SleepSchedule == "6 hours":
            if age < 18:
                myhtml = f"""
                <h1> 8 hours! </h1>
                <p> Great job <b>{name}</b>! Continue to do what you are doing. Stick to this sleep schedule and you will be great. </p>"""
                report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General ways to improve sleep")
                    report.save("report_add_html.html", overwrite=True)
            elif age == 18:
                myhtml = f"""
                <h1> Sleep Recommendations </h1>
                <p> You're still growing {name}, so make sure to stay consistent and endure the growth.</p>
                """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General ways to improve sleep")
                    report.save("report_add_html.html", overwrite=True)
            elif age > 18 and age < 27:
                myhtml = f"""
                <h1> {name}'s Sleep Recommendations </h1>
                <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
                """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General ways to improve sleep")
                    report.save("report_add_html.html", overwrite=True)
            elif age > 30 and age < 45:
                myhtml = f"""
                <h1> {name}'s Sleep Recommendations </h1>
                <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
                """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General ways to improve sleep")
                    report.save("report_add_html.html", overwrite=True)
            elif age > "55" and age < "70":
                myhtml = """
                <h1> At your age, the main goal would be to aim for more sleep </h1>
                """
                report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General ways to improve sleep")
                    report.save("report_add_html.html", overwrite=True)
        elif SleepSchedule == "5 hours":
            if age < 18:
                myhtml = f"""
                <h1> 8 hours! </h1>
                <p> Great job <b>{name}</b>! Continue to do what you are doing. Stick to this sleep schedule and you will be great. </p>"""
                report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General ways to improve sleep")
                    report.save("report_add_html.html", overwrite=True)
            elif age == 18:
                myhtml = f"""
                <h1> Sleep Recommendations </h1>
                <p> You're still growing {name}, so make sure to stay consistent and endure the growth.</p>
                """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General ways to improve sleep")
                    report.save("report_add_html.html", overwrite=True)
            elif age > 18 and age < 27:
                myhtml = f"""
                <h1> {name}'s Sleep Recommendations </h1>
                <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
                """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General ways to improve sleep")
                    report.save("report_add_html.html", overwrite=True)
            elif age > 30 and age < 45:
                myhtml = f"""
                <h1> {name}'s Sleep Recommendations </h1>
                <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
                """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General ways to improve sleep")
                    report.save("report_add_html.html", overwrite=True)
            elif age > "55" and age < "70":
                myhtml = """
                <h1> At your age, the main goal would be to aim for more sleep </h1>
                """
                report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General ways to improve sleep")
                    report.save("report_add_html.html", overwrite=True)

        elif SleepSchedule == "11 hours":
            if age < 18:
                myhtml = f"""
                <h1> 8 hours! </h1>
                <p> Great job <b>{name}</b>! Continue to do what you are doing. Stick to this sleep schedule and you will be great. </p>"""
                report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General ways to improve sleep")
                    report.save("report_add_html.html", overwrite=True)
            elif age == 18:
                myhtml = f"""
                <h1> Sleep Recommendations </h1>
                <p> You're still growing {name}, so make sure to stay consistent and endure the growth.</p>
                """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General ways to improve sleep")
                    report.save("report_add_html.html", overwrite=True)
            elif age > 18 and age < 27:
                myhtml = f"""
                <h1> {name}'s Sleep Recommendations </h1>
                <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
                """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General ways to improve sleep")
                    report.save("report_add_html.html", overwrite=True)
            elif age > 30 and age < 45:
                myhtml = f"""
                <h1> {name}'s Sleep Recommendations </h1>
                <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
                """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General ways to improve sleep")
                    report.save("report_add_html.html", overwrite=True)
            elif age > "55" and age < "70":
                myhtml = """
                <h1> At your age, the main goal would be to aim for more sleep </h1>
                """
                report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General ways to improve sleep")
                    report.save("report_add_html.html", overwrite=True)
    elif melatonin_get == "25":
        myhtml = f"""
            <h1> {name}'s Personalized Sleep Recommendations </h1>
            <p>
            Through computed <b>melatonin</b> levels, here's what we suggest you do:
            <p></p>
            <p>
            1. Make sure to stay hydrated
            </p>
            <p></p>
            <p>
            2. Make sure to limit the exposure to blue light before you sleep
            </p>
            <p></p>
            <p>
            3. Make sure to drop your body temperature before you sleep
            </p>
            """
        report.add_html(html=myhtml, title=name + "'s Personalized Recommendations")
        report.save("report_add_html.html", overwrite=True)
        if SleepSchedule == "10 hours":
            if age < 18:
                myhtml = f"""
                <h1> Great Job {name}! </h1>
                <p> This will help with sleep immensely. Stay consistent and continue to endure growth and development. </p>
                """
                report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)

            elif age == 18:
                myhtml = """
                    <h1> Great Job </h1>
                    <p> This will help with sleep immensely. Stay consistent and continue to endure growth and development. </p>
                    """
                report.add_html(html=myhtml, title="data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)

            elif age > 18 and age < 27:
                myhtml = """
                <h1> Keep Doing What You're Doing </h1>
                <p> 10 hours is probably a little too much, but it helps with growth. </p>
                <p></p>
                <p> As you'll be wrapping up your growth within the ages of 18-27, this is the time where you shouldn't compromise your growth. Keep doing what you're doing. </p>
                """
                report.add_html(html=myhtml, title="data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > 30 and age < 45:
                myhtml = """
                <h1> Wow </h1>
                <p> That's a great amount of sleep. Great job!
                """
                report.add_html(html=myhtml, title="data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > "55" and age < "70":
                myhtml = """
                    <h1> At your age, the main goal would be to aim for more sleep </h1>
                    """
                report.add_html(html=myhtml, title="data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
        elif SleepSchedule == "9 hours":
            if age == 18:
                myhtml = """
                    <h1> Good Job </h1>
                    <p> This will help with sleep immensely. Stay consistent and continue to endure growth and development. </p>
                    """
                report.add_html(html=myhtml, title="data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > "18" and age < "27":
                myhtml = """
                <h1> Keep Doing What You're Doing </h1>
                <p> 9 hours is a great amount of sleep </p>
                <p></p>
                <p> As you'll be wrapping up your growth within the ages of 18-27, this is the time where you shouldn't compromise your growth. Keep doing what you're doing. </p>
                """
                report.add_html(html=myhtml, title="data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > "30" and age < "45":
                myhtml = """
                <h1> Wow </h1>
                <p> That's a good amount of sleep. Great job!
                """
                report.add_html(html=myhtml, title="data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > "55" and age < "70":
                myhtml = """
                    <h1> At your age, the main goal would be to aim for more sleep </h1>
                    """
                report.add_html(html=myhtml, title="data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
        elif SleepSchedule == "8 hours":
            if age < 18:
                myhtml = f"""
                <h1> 8 hours! </h1>
                <p> Great job <b>{name}</b>! Continue to do what you are doing. Stick to this sleep schedule and you will be great. </p>"""
                report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age == 18:
                myhtml = f"""
                <h1> Sleep Recommendations </h1>
                <p> You're still growing {name}, so make sure to stay consistent and endure the growth.</p>
                """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > 18 and age < 27:
                myhtml = f"""
                <h1> {name}'s Sleep Recommendations </h1>
                <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
                """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > 30 and age < 45:
                myhtml = f"""
                <h1> {name}'s Sleep Recommendations </h1>
                <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
                """
                report.add_html(html=myhtml, title="Data")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="General Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
            elif age > "55" and age < "70":
                myhtml = """
                <h1> At your age, the main goal would be to aim for more sleep </h1>
                """
                report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
                report.save("report_add_html.html", overwrite=True)
                if specificity == "1":
                    myhtml = f"""
                    <h1> Here are specific ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                    <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                    <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                    <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                    report.add_html(html=myhtml, title="Specific Recommendation Steps")
                    report.save("report_add_html.html", overwrite=True)
                elif specificity == "2":
                    myhtml = f"""
                    <h1> Here are GENERAL ways to conduct your recommendations </h1>
                    <h2> 1. Blue Light Emissions </h2>
                    <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                    <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                    <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                    <h2> 2. Caffeine </h2>
                    <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                    <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                    <h2> 3. Sleep Habits </h2>
                    <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
    elif SleepSchedule == "7 hours":
        if age < 18:
            myhtml = f"""
            <h1> 8 hours! </h1>
            <p> Great job <b>{name}</b>! Continue to do what you are doing. Stick to this sleep schedule and you will be great. </p>"""
            report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age == 18:
            myhtml = f"""
            <h1> Sleep Recommendations </h1>
            <p> You're still growing {name}, so make sure to stay consistent and endure the growth.</p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 18 and age < 27:
            myhtml = f"""
            <h1> {name}'s Sleep Recommendations </h1>
            <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 30 and age < 45:
            myhtml = f"""
            <h1> {name}'s Sleep Recommendations </h1>
            <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > "55" and age < "70":
            myhtml = """
            <h1> At your age, the main goal would be to aim for more sleep </h1>
            """
            report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)

            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)

    elif SleepSchedule == "6 hours":
        if age < 18:
            myhtml = f"""
            <h1> 8 hours! </h1>
            <p> Great job <b>{name}</b>! Continue to do what you are doing. Stick to this sleep schedule and you will be great. </p>"""
            report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age == 18:
            myhtml = f"""
            <h1> Sleep Recommendations </h1>
            <p> You're still growing {name}, so make sure to stay consistent and endure the growth.</p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 18 and age < 27:
            myhtml = f"""
            <h1> {name}'s Sleep Recommendations </h1>
            <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 30 and age < 45:
            myhtml = f"""
            <h1> {name}'s Sleep Recommendations </h1>
            <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > "55" and age < "70":
            myhtml = """
            <h1> At your age, the main goal would be to aim for more sleep </h1>
            """
            report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
    elif SleepSchedule == "5 hours":
        if age < 18:
            myhtml = f"""
            <h1> 8 hours! </h1>
            <p> Great job <b>{name}</b>! Continue to do what you are doing. Stick to this sleep schedule and you will be great. </p>"""
            report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age == 18:
            myhtml = f"""
            <h1> Sleep Recommendations </h1>
            <p> You're still growing {name}, so make sure to stay consistent and endure the growth.</p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 18 and age < 27:
            myhtml = f"""
            <h1> {name}'s Sleep Recommendations </h1>
            <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 30 and age < 45:
            myhtml = f"""
            <h1> {name}'s Sleep Recommendations </h1>
            <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > "55" and age < "70":
            myhtml = """
            <h1> At your age, the main goal would be to aim for more sleep </h1>
            """
            report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)

    elif SleepSchedule == "11 hours":
        if age < 18:
            myhtml = f"""
            <h1> 8 hours! </h1>
            <p> Great job <b>{name}</b>! Continue to do what you are doing. Stick to this sleep schedule and you will be great. </p>"""
            report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age == 18:
            myhtml = f"""
            <h1> Sleep Recommendations </h1>
            <p> You're still growing {name}, so make sure to stay consistent and endure the growth.</p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 18 and age < 27:
            myhtml = f"""
            <h1> {name}'s Sleep Recommendations </h1>
            <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 30 and age < 45:
            myhtml = f"""
            <h1> {name}'s Sleep Recommendations </h1>
            <p> Main thing is to stay consistent. Also, make sure to practice good routine practices when you go to sleep. </p>
            """
            report.add_html(html=myhtml, title="Data")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
        elif age > 55 and age < 70:
            myhtml = """
            <h1> At your age, the main goal would be to aim for more sleep </h1>
            """
            report.add_html(html=myhtml, title=name + "'s Personalized Sleep Recommendations")
            report.save("report_add_html.html", overwrite=True)
            if specificity == "1":
                myhtml = f"""
                <h1> Here are specific ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="Specific Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
            elif specificity == "2":
                myhtml = f"""
                <h1> Here are GENERAL ways to conduct your recommendations </h1>
                <h2> 1. Blue Light Emissions </h2>
                <p> - Make sure to turn your phone to night shift mode so that you can limit the amount of blue light emissions that come to your eyes. </p>
                <p> - Make sure to not look at screens for at least 15-30 minutes before you sleep. </p>
                <p> - Implement a sleeping habit, such as journaling, so that you don't look at your phone right before you sleep. </p>
                <h2> 2. Caffeine </h2>
                <p> - Make sure to drink caffeine before 2pm. Here's the context: Caffeine is very alike to a molecule called adenosine. When taking caffeine, the molecules can replace the receptors that are meant for adenosine. This can affect the process as you can feel awake during the night, affecting your ability to sleep. </p>
                <p> - Only consume a moderate amount of caffeine. Take about 50-100 mg of caffeine for adequate production and normalized sleep. </p>
                <h2> 3. Sleep Habits </h2>
                <p> You want to be able to set sleep habits that make you feel better and end up helping your performance. Here's how you do so. </p>
                <p> - Set up a bedtime habit. This involves getting yourself prepared for sleep, preparing yourself for the next day etc. </p>
                <p> - Try to conduct habits that limit your phone usage before you sleep. This involves reading a book, journaling, preparing things for the next day, or just exposing yourself to the darkness to help your eyes adjust to the darker light. </p>
                <p> <i> If there are any concerns, send feedback about the recommendations to <b> abhizgunna@gmail.com </b> </i> </p>"""

                report.add_html(html=myhtml, title="General Recommendation Steps")
                report.save("report_add_html.html", overwrite=True)
    else:
        # Only if melatonin values seem to be compromised. This usually won't be the case.
        myhtml = """
            <h1>I'm not sure about this response</h1>
            <p>Consider running the system again. If it fails again, send feedback to the creator at <i> abhizgunna@gmail.com </i> </p>"""
        report.add_html(html=myhtml, title="Error")
        report.save("report_add_html.html", overwrite=True)
        sys.exit()

    # Assuming 'input_name' is the name attribute of your input field
    # Process the data here
    return 'Thank you.'


#report.save("report_add_html.html", overwrite=True)
#print(f"""
#{name}'s data report has been generated!
#""")


if __name__ == '__main__':
    #app.run(debug=True, port=5002)
    serve(app, host='0.0.0.0', port=5004, threads=4)
